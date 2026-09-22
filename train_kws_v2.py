import os
import json
import random
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

#config

DATASET_ROOT = Path("datasetc")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

#ESP sets to be identical
N_MFCC = 13
N_MELS = 40
N_FFT = 2048
HOP_LENGTH = 1024
FMIN = 20
FMAX = SAMPLE_RATE // 2
CENTER = False

#Training
TRAIN_NEGATIVE_STRIDE_SEC = 0.50
EVAL_NEGATIVE_STRIDE_SEC = 1.00
MAX_NEGATIVE_TO_POSITIVE = 2.0

#For positive keyword recordings longer than 1 second
KEYWORD_JITTERS_SEC = (-0.15, 0.0, 0.15)

#Training augmentation
AUGMENT_COPIES = 1
AUGMENT_GAIN_DB = (-6.0, 6.0)
AUGMENT_SHIFT_SEC = (-0.10, 0.10)
AUGMENT_NOISE_PROB = 0.70
AUGMENT_NOISE_SNR_DB = (5.0, 20.0)

#Threshold selection 
TARGET_RECALL = 0.95

#Model
EPOCHS = 60
BATCH_SIZE = 32


#utils

def list_wavs(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    )


def load_audio(path: Path) -> np.ndarray:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"Unexpected sample rate for {path}: {sr}")
    y = y.astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def pad_or_crop_center(y: np.ndarray) -> np.ndarray:
    """Make exactly 1 second without removing silence from the chosen window."""
    if len(y) == CLIP_SAMPLES:
        return y.copy()

    if len(y) < CLIP_SAMPLES:
        total = CLIP_SAMPLES - len(y)
        left = total // 2
        right = total - left
        return np.pad(y, (left, right), mode="constant")

    start = (len(y) - CLIP_SAMPLES) // 2
    return y[start:start + CLIP_SAMPLES].copy()


def make_negative_windows(y: np.ndarray, stride_sec: float):
    """Segment a long negative/background recording into fixed 1-second windows."""
    stride = max(1, int(stride_sec * SAMPLE_RATE))

    if len(y) <= CLIP_SAMPLES:
        return [pad_or_crop_center(y)]

    windows = []
    start = 0
    while start + CLIP_SAMPLES <= len(y):
        windows.append(y[start:start + CLIP_SAMPLES].copy())
        start += stride

    #tail so the end of a recording is not ignored
    if start < len(y):
        tail = y[-CLIP_SAMPLES:]
        windows.append(tail.copy())

    return windows


def make_keyword_windows(y: np.ndarray, training: bool):
    """
    Find where speech is located only to select a useful crop.
    The resulting crop itself is NOT trimmed: surrounding silence remains.
    """
    if len(y) <= CLIP_SAMPLES:
        return [pad_or_crop_center(y)]

    try:
        intervals = librosa.effects.split(
            y,
            top_db=35,
            frame_length=512,
            hop_length=128,
        )
    except Exception:
        intervals = np.empty((0, 2), dtype=np.int64)

    if len(intervals) > 0:
        speech_start = intervals[:, 0].min()
        speech_end = intervals[:, 1].max()
        center = (speech_start + speech_end) / 2.0
    else:
        center = len(y) / 2.0

    base_start = int(round(center - CLIP_SAMPLES / 2))

    def crop_at(start):
        start = max(0, min(start, len(y) - CLIP_SAMPLES))
        return y[start:start + CLIP_SAMPLES].copy()

    if not training:
        return [crop_at(base_start)]

    windows = []
    for jitter_sec in KEYWORD_JITTERS_SEC:
        jitter = int(round(jitter_sec * SAMPLE_RATE))
        windows.append(crop_at(base_start + jitter))

    return windows


#MFCC extraction

def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """
    Streaming-compatible MFCC:
      - fixed 16 kHz
      - 13 MFCC coefficients
      - 2048-sample FFT
      - 1024-sample hop
      - center=False

    The ESP32 must reproduce these settings.
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann",
        center=CENTER,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    return mfcc.T.astype(np.float32)


#dataset loading

def load_split_waveforms(split_name: str, training: bool):

    split_dir = DATASET_ROOT / split_name
    keyword_dir = split_dir / "keyword"
    negative_dir = split_dir / "negative"

    if not keyword_dir.exists():
        raise FileNotFoundError(f"Missing: {keyword_dir}")
    if not negative_dir.exists():
        raise FileNotFoundError(f"Missing: {negative_dir}")

    keyword_files = list_wavs(keyword_dir)
    negative_files = list_wavs(negative_dir)

    if not keyword_files:
        raise RuntimeError(f"No .wav files found under {keyword_dir}")
    if not negative_files:
        raise RuntimeError(f"No .wav files found under {negative_dir}")

    keyword_windows = []
    negative_windows = []

    print(f"\n[{split_name.upper()}]")
    print(f"Keyword files:  {len(keyword_files)}")
    print(f"Negative files: {len(negative_files)}")

    for path in keyword_files:
        y = load_audio(path)
        keyword_windows.extend(
            make_keyword_windows(y, training=training)
        )

    stride = (
        TRAIN_NEGATIVE_STRIDE_SEC
        if training
        else EVAL_NEGATIVE_STRIDE_SEC
    )

    for path in negative_files:
        y = load_audio(path)
        negative_windows.extend(
            make_negative_windows(y, stride_sec=stride)
        )

    print(f"Keyword windows before cap:  {len(keyword_windows)}")
    print(f"Negative windows before cap: {len(negative_windows)}")

    return keyword_windows, negative_windows


#augmentation   

def time_shift(y: np.ndarray, shift_samples: int) -> np.ndarray:
    out = np.zeros_like(y)

    if shift_samples > 0:
        out[shift_samples:] = y[:-shift_samples]
    elif shift_samples < 0:
        out[:shift_samples] = y[-shift_samples:]
    else:
        out[:] = y

    return out


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)) + 1e-8))


def mix_background(signal: np.ndarray, noise: np.ndarray, snr_db: float):
    if len(noise) != len(signal):
        noise = pad_or_crop_center(noise)

    signal_rms = rms(signal)
    noise_rms = rms(noise)

    if noise_rms < 1e-6:
        return signal.copy()

    desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (desired_noise_rms / noise_rms)

    mixed = signal + scaled_noise
    return np.clip(mixed, -1.0, 1.0)


def augment_waveform(
    y: np.ndarray,
    label: int,
    noise_pool,
    rng: np.random.Generator,
):
    out = y.copy()

    # Random gain
    gain_db = rng.uniform(*AUGMENT_GAIN_DB)
    gain = 10.0 ** (gain_db / 20.0)
    out *= gain

    # Small time shift
    shift_sec = rng.uniform(*AUGMENT_SHIFT_SEC)
    shift_samples = int(round(shift_sec * SAMPLE_RATE))
    out = time_shift(out, shift_samples)

    # Background mixing
    if noise_pool and rng.random() < AUGMENT_NOISE_PROB:
        noise = noise_pool[
            int(rng.integers(0, len(noise_pool)))
        ]
        snr_db = rng.uniform(*AUGMENT_NOISE_SNR_DB)
        out = mix_background(out, noise, snr_db)

    # Small sensor-like Gaussian noise
    if rng.random() < 0.30:
        noise_std = rng.uniform(0.0005, 0.003)
        out = out + rng.normal(0.0, noise_std, size=out.shape)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def build_training_waves():
    keyword_waves, negative_waves = load_split_waveforms(
        "train",
        training=True,
    )

    rng = np.random.default_rng(SEED)
    rng.shuffle(keyword_waves)
    rng.shuffle(negative_waves)

    #Prevent a huge noise collection from overwhelming the keyword class.
    max_negative = max(
        1,
        int(len(keyword_waves) * MAX_NEGATIVE_TO_POSITIVE)
    )

    if len(negative_waves) > max_negative:
        idx = rng.choice(
            len(negative_waves),
            size=max_negative,
            replace=False,
        )
        negative_waves = [negative_waves[i] for i in idx]

    print(f"\nTRAIN after negative cap:")
    print(f"Keyword windows:  {len(keyword_waves)}")
    print(f"Negative windows: {len(negative_waves)}")

    # Noise pool is kept from TRAIN only.
    noise_pool = negative_waves.copy()

    waves = [(w, 1) for w in keyword_waves]
    waves += [(w, 0) for w in negative_waves]

    rng.shuffle(waves)

    # Add augmented copies.
    if AUGMENT_COPIES > 0:
        original = waves.copy()

        for _ in range(AUGMENT_COPIES):
            for wave, label in original:
                aug = augment_waveform(
                    wave,
                    label,
                    noise_pool,
                    rng,
                )
                waves.append((aug, label))

    rng.shuffle(waves)

    return waves


def waves_to_features(waves):
    X = np.stack(
        [extract_mfcc(w) for w, _ in waves]
    ).astype(np.float32)

    y = np.array(
        [label for _, label in waves],
        dtype=np.int32,
    )

    return X, y


#normalization

def normalize_train_test(
    X_train_raw,
    X_val_raw,
    X_test_raw,
):
    mean = X_train_raw.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = X_train_raw.std(
        axis=(0, 1),
        keepdims=True,
    ) + 1e-8

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    return X_train, X_val, X_test, mean, std


#model

def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.SeparableConv2D(
        8,
        (3, 3),
        padding="same",
        activation="relu",
    )(inputs)

    x = keras.layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = keras.layers.SeparableConv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu",
    )(x)

    x = keras.layers.MaxPooling2D(
        (2, 2)
    )(x)

    # Smaller and more robust than flattening the whole feature map.
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(
        16,
        activation="relu",
    )(x)

    x = keras.layers.Dropout(0.20)(x)

    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="tiny_dscnn_kws",
    )


#threshold selection

def choose_threshold(y_val, val_probs):
    candidates = np.linspace(0.05, 0.99, 95)

    valid = []

    for threshold in candidates:
        preds = (val_probs >= threshold).astype(np.int32)

        recall = recall_score(
            y_val,
            preds,
            zero_division=0,
        )

        precision = precision_score(
            y_val,
            preds,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            preds,
            zero_division=0,
        )

        negatives = (y_val == 0)
        fp = np.sum((preds == 1) & negatives)
        tn = np.sum((preds == 0) & negatives)
        fpr = fp / max(1, fp + tn)

        if recall >= TARGET_RECALL:
            valid.append(
                (
                    fpr,
                    -precision,
                    -f1,
                    -threshold,
                    threshold,
                    recall,
                    precision,
                    f1,
                )
            )

    if valid:
        valid.sort()
        best = valid[0]
        return {
            "threshold": float(best[4]),
            "recall": float(best[5]),
            "precision": float(best[6]),
            "f1": float(best[7]),
            "selection_rule": (
                f"minimum validation FPR subject to recall >= "
                f"{TARGET_RECALL:.2f}"
            ),
        }

    # Fallback: maximize F1 if target recall is unreachable.
    best = None
    for threshold in candidates:
        preds = (val_probs >= threshold).astype(np.int32)

        f1 = f1_score(
            y_val,
            preds,
            zero_division=0,
        )

        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "recall": float(
                    recall_score(
                        y_val,
                        preds,
                        zero_division=0,
                    )
                ),
                "precision": float(
                    precision_score(
                        y_val,
                        preds,
                        zero_division=0,
                    )
                ),
                "f1": float(f1),
                "selection_rule": "maximum validation F1",
            }

    return best


#TFLite INT8 Quantization

def representative_dataset_generator(X_train):
    rng = np.random.default_rng(SEED)

    count = min(300, len(X_train))
    indices = rng.choice(
        len(X_train),
        size=count,
        replace=False,
    )

    for idx in indices:
        yield [
            X_train[idx:idx + 1].astype(np.float32)
        ]


def quantize_model(model, X_train):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        lambda: representative_dataset_generator(X_train)
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def int8_predict(model_bytes, X):
    interpreter = tf.lite.Interpreter(
        model_content=model_bytes
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    if in_scale == 0 or out_scale == 0:
        raise RuntimeError(
            "Invalid INT8 quantization scale."
        )

    probs = []

    for x in X:
        q = np.round(
            x / in_scale + in_zero
        )

        q = np.clip(
            q,
            -128,
            127,
        ).astype(np.int8)

        interpreter.set_tensor(
            input_details["index"],
            q[np.newaxis, ...],
        )

        interpreter.invoke()

        raw = interpreter.get_tensor(
            output_details["index"]
        )[0][0]

        probability = (
            raw - out_zero
        ) * out_scale

        probs.append(float(probability))

    return np.array(probs, dtype=np.float32), {
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zero),
        "output_scale": float(out_scale),
        "output_zero_point": int(out_zero),
    }


#export headers

def export_c_array(data_bytes, array_name, header_path):
    data = bytes(data_bytes)

    with open(header_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"alignas(8) const unsigned char {array_name}[] = {{\n")

        for i in range(0, len(data), 20):
            chunk = data[i:i + 20]
            line = ", ".join(
                f"0x{b:02x}" for b in chunk
            )
            f.write(f"    {line},\n")

        f.write("};\n")
        f.write(
            f"const unsigned int {array_name}_len = "
            f"{len(data)};\n"
        )


def export_normalization_header(mean, std):
    mean_flat = mean.flatten()
    std_flat = std.flatten()

    path = ARTIFACT_DIR / "normalization.h"

    with open(path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <cstddef>\n\n")

        f.write(
            f"constexpr int MFCC_NUM_COEFFS = "
            f"{len(mean_flat)};\n\n"
        )

        f.write(
            "alignas(4) constexpr float MFCC_MEAN[] = {"
            + ", ".join(
                f"{v:.9f}f" for v in mean_flat
            )
            + "};\n"
        )

        f.write(
            "alignas(4) constexpr float MFCC_STD[] = {"
            + ", ".join(
                f"{v:.9f}f" for v in std_flat
            )
            + "};\n"
        )


def export_config_header():
    path = ARTIFACT_DIR / "kws_config.h"

    with open(path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n\n")
        f.write(f"constexpr int KWS_SAMPLE_RATE = {SAMPLE_RATE};\n")
        f.write(f"constexpr int KWS_CLIP_SAMPLES = {CLIP_SAMPLES};\n")
        f.write(f"constexpr int KWS_N_MFCC = {N_MFCC};\n")
        f.write(f"constexpr int KWS_N_MELS = {N_MELS};\n")
        f.write(f"constexpr int KWS_N_FFT = {N_FFT};\n")
        f.write(f"constexpr int KWS_HOP_LENGTH = {HOP_LENGTH};\n")
        f.write(f"constexpr int KWS_FMIN = {FMIN};\n")
        f.write(f"constexpr int KWS_FMAX = {FMAX};\n")
        f.write(
            f"constexpr bool KWS_CENTER = "
            f"{str(CENTER).lower()};\n"
        )


#main

def main():
    print("====================================================")
    print(" TinyML KWS training pipeline")
    print("====================================================")
    print(f"Sample rate : {SAMPLE_RATE}")
    print(f"Clip        : {CLIP_SECONDS:.1f} s")
    print(f"MFCC        : {N_MFCC}")
    print(f"N_FFT       : {N_FFT}")
    print(f"HOP_LENGTH  : {HOP_LENGTH}")
    print(f"CENTER      : {CENTER}")

    # -------------------------
    # Load training
    # -------------------------
    train_waves = build_training_waves()

    # -------------------------
    # Load validation/test
    # -------------------------
    val_kw, val_neg = load_split_waveforms(
        "val",
        training=False,
    )

    test_kw, test_neg = load_split_waveforms(
        "test",
        training=False,
    )

    val_waves = (
        [(w, 1) for w in val_kw]
        + [(w, 0) for w in val_neg]
    )

    test_waves = (
        [(w, 1) for w in test_kw]
        + [(w, 0) for w in test_neg]
    )

    # -------------------------
    # Feature extraction
    # -------------------------
    print("\nExtracting MFCCs...")

    X_train_raw, y_train = waves_to_features(
        train_waves
    )

    X_val_raw, y_val = waves_to_features(
        val_waves
    )

    X_test_raw, y_test = waves_to_features(
        test_waves
    )

    print("Raw shapes:")
    print("  train:", X_train_raw.shape)
    print("  val:  ", X_val_raw.shape)
    print("  test: ", X_test_raw.shape)

    # -------------------------
    # Normalize from TRAIN only
    # -------------------------
    (
        X_train,
        X_val,
        X_test,
        mfcc_mean,
        mfcc_std,
    ) = normalize_train_test(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )

    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    input_shape = X_train.shape[1:]

    # -------------------------
    # Model
    # -------------------------
    print("\nBuilding model...")
    model = build_model(input_shape)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            ARTIFACT_DIR / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # -------------------------
    # Float validation
    # -------------------------
    val_probs = model.predict(
        X_val,
        verbose=0,
    ).ravel()

    threshold_info = choose_threshold(
        y_val,
        val_probs,
    )

    threshold = threshold_info["threshold"]

    print("\nChosen threshold:")
    print(json.dumps(
        threshold_info,
        indent=2,
    ))

    # -------------------------
    # Float test
    # -------------------------
    test_probs = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    test_preds = (
        test_probs >= threshold
    ).astype(np.int32)

    print("\n========== FLOAT TEST ==========")
    print(
        classification_report(
            y_test,
            test_preds,
            target_names=[
                "negative",
                "keyword",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            test_preds,
        )
    )

    # Window-level negative false accepts.
    test_negative_count = int(
        np.sum(y_test == 0)
    )

    false_accepts = int(
        np.sum(
            (y_test == 0)
            & (test_preds == 1)
        )
    )

    if test_negative_count > 0:
        negative_hours = (
            test_negative_count
            * EVAL_NEGATIVE_STRIDE_SEC
            / 3600.0
        )

        fa_per_hour = (
            false_accepts / negative_hours
            if negative_hours > 0
            else float("inf")
        )
    else:
        fa_per_hour = None

    #int8 quantization
    print("\nQuantizing to full INT8...")
    tflite_quant_model = quantize_model(
        model,
        X_train,
    )

    tflite_path = (
        ARTIFACT_DIR / "micro_kws_model.tflite"
    )

    with open(
        tflite_path,
        "wb",
    ) as f:
        f.write(tflite_quant_model)

   # int8 test
    q_test_probs, quant_info = int8_predict(
        tflite_quant_model,
        X_test,
    )

    q_test_preds = (
        q_test_probs >= threshold
    ).astype(np.int32)

    print("\n========== INT8 TEST ==========")
    print(
        classification_report(
            y_test,
            q_test_preds,
            target_names=[
                "negative",
                "keyword",
            ],
            digits=4,
            zero_division=0,
        )
    )

    #export headers
    export_c_array(
        tflite_quant_model,
        "model_tflite",
        ARTIFACT_DIR / "model.h",
    )

    export_normalization_header(
        mfcc_mean,
        mfcc_std,
    )

    export_config_header()

    #metadata and metrics
    final_metadata = {
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "clip_samples": CLIP_SAMPLES,
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "fmin": FMIN,
        "fmax": FMAX,
        "center": CENTER,
        "threshold": threshold,
        "threshold_rule": threshold_info["selection_rule"],
        "float_test_accuracy": float(
            np.mean(test_preds == y_test)
        ),
        "float_test_precision": float(
            precision_score(
                y_test,
                test_preds,
                zero_division=0,
            )
        ),
        "float_test_recall": float(
            recall_score(
                y_test,
                test_preds,
                zero_division=0,
            )
        ),
        "float_test_f1": float(
            f1_score(
                y_test,
                test_preds,
                zero_division=0,
            )
        ),
        "int8_test_accuracy": float(
            np.mean(q_test_preds == y_test)
        ),
        "int8_test_precision": float(
            precision_score(
                y_test,
                q_test_preds,
                zero_division=0,
            )
        ),
        "int8_test_recall": float(
            recall_score(
                y_test,
                q_test_preds,
                zero_division=0,
            )
        ),
        "int8_test_f1": float(
            f1_score(
                y_test,
                q_test_preds,
                zero_division=0,
            )
        ),
        "test_false_accepts": false_accepts,
        "test_negative_windows": test_negative_count,
        "window_level_false_accepts_per_hour": fa_per_hour,
        "quantization": quant_info,
    }

    with open(
        ARTIFACT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_metadata,
            f,
            indent=2,
        )

    model.save(
        ARTIFACT_DIR / "kws_model.keras"
    )

    print("\n====================================================")
    print("DONE")
    print("====================================================")
    print(f"Artifacts written to: {ARTIFACT_DIR.resolve()}")
    print(" - kws_model.keras")
    print(" - best_model.keras")
    print(" - micro_kws_model.tflite")
    print(" - model.h")
    print(" - normalization.h")
    print(" - kws_config.h")
    print(" - metrics.json")
    print("\nIMPORTANT:")
    print("Use the exact MFCC settings in kws_config.h on the ESP32.")
    print("Do not use librosa.effects.trim() in the ESP32 pipeline.")
    print("Keep all recordings from one speaker/file in only one split.")


if __name__ == "__main__":
    main()
