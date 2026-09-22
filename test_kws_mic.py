import argparse
import json
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Install sounddevice with: pip install sounddevice")
    raise

try:
    import tensorflow as tf
except ImportError:
    print("Install TensorFlow with: pip install tensorflow")
    raise

# Must match train_kws_v2.py
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

N_MFCC = 13
N_MELS = 40
N_FFT = 2048
HOP_LENGTH = 1024
FMIN = 20
FMAX = SAMPLE_RATE // 2
CENTER = False

DEFAULT_MODEL = Path("artifacts/micro_kws_model.tflite")
DEFAULT_NORM = Path("artifacts/normalization.h")
DEFAULT_METRICS = Path("artifacts/metrics.json")


def parse_c_float_array(header_text, array_name):
    pattern = rf"{re.escape(array_name)}\[\]\s*=\s*\{{(.*?)\}}\s*;"
    match = re.search(pattern, header_text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find {array_name}[] in normalization.h")

    values = []
    for token in match.group(1).split(","):
        token = token.strip()
        if not token:
            continue
        token = token.rstrip("fF")
        values.append(float(token))

    return np.asarray(values, dtype=np.float32)


def load_normalization(path):
    text = path.read_text(encoding="utf-8")
    mean = parse_c_float_array(text, "MFCC_MEAN")
    std = parse_c_float_array(text, "MFCC_STD")

    if len(mean) != N_MFCC or len(std) != N_MFCC:
        raise RuntimeError(
            f"Expected {N_MFCC} MFCC normalization values; "
            f"got mean={len(mean)}, std={len(std)}."
        )

    return mean.reshape(1, N_MFCC), std.reshape(1, N_MFCC)


def extract_mfcc(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < CLIP_SAMPLES:
        padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        start = (CLIP_SAMPLES - len(audio)) // 2
        padded[start:start + len(audio)] = audio
        audio = padded
    elif len(audio) > CLIP_SAMPLES:
        audio = audio[-CLIP_SAMPLES:]

    mfcc = librosa.feature.mfcc(
        y=audio,
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


class QuantizedKWS:
    def __init__(self, model_path, mean, std):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.mean = mean
        self.std = std

        self.in_scale, self.in_zero = self.input_details["quantization"]
        self.out_scale, self.out_zero = self.output_details["quantization"]

        if self.in_scale == 0 or self.out_scale == 0:
            raise RuntimeError("Invalid INT8 quantization scale.")

        print("Model:")
        print("  input shape:", tuple(self.input_details["shape"]))
        print("  input dtype:", self.input_details["dtype"])
        print("  input quantization:", self.in_scale, self.in_zero)
        print("  output dtype:", self.output_details["dtype"])
        print("  output quantization:", self.out_scale, self.out_zero)

    def predict(self, audio):
        mfcc = extract_mfcc(audio)
        normalized = (mfcc - self.mean) / self.std
        x = normalized[np.newaxis, ..., np.newaxis]

        expected = tuple(int(v) for v in self.input_details["shape"])
        if tuple(x.shape) != expected:
            raise RuntimeError(
                f"Feature shape {tuple(x.shape)} != model input {expected}. "
                "Training and microphone preprocessing do not match."
            )

        q = np.round(x / self.in_scale + self.in_zero)
        q = np.clip(q, -128, 127).astype(self.input_details["dtype"])

        self.interpreter.set_tensor(self.input_details["index"], q)
        self.interpreter.invoke()

        raw = self.interpreter.get_tensor(self.output_details["index"])
        prob = (float(raw.reshape(-1)[0]) - self.out_zero) * self.out_scale
        return float(np.clip(prob, 0.0, 1.0))


def load_threshold(metrics_path, fallback):
    if not metrics_path.exists():
        return fallback
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(data["threshold"])
        if 0.0 < threshold < 1.0:
            return threshold
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Test TinyML KWS with laptop microphone")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORM)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--hits", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--show-db", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not 0 < args.interval <= CLIP_SECONDS:
        raise ValueError("--interval must be > 0 and <= 1.0")
    if args.hits < 1:
        raise ValueError("--hits must be >= 1")

    for p in (args.model, args.normalization):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    mean, std = load_normalization(args.normalization)
    threshold = (
        args.threshold
        if args.threshold is not None
        else load_threshold(args.metrics, 0.85)
    )

    kws = QuantizedKWS(args.model, mean, std)

    block_size = max(1, int(round(args.interval * SAMPLE_RATE)))
    ring = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    recent_hits = []
    last_detection = -float("inf")

    print("\nAudio configuration:")
    print(f"  sample rate : {SAMPLE_RATE} Hz")
    print(f"  window      : {CLIP_SECONDS:.2f} s")
    print(f"  interval    : {args.interval:.2f} s")
    print(f"  threshold   : {threshold:.4f}")
    print(f"  required hits: {args.hits}")
    print("\nSpeak the wake word. Ctrl+C to stop.")

    def callback(indata, frames, time_info, status):
        nonlocal ring
        if status:
            print(f"\nAudio status: {status}", file=sys.stderr)
        samples = indata[:, 0].astype(np.float32)
        if len(samples) >= len(ring):
            ring = samples[-len(ring):].copy()
        else:
            ring = np.concatenate((ring[len(samples):], samples))

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            device=args.device,
            callback=callback,
        ):
            while True:
                time.sleep(args.interval)
                window = ring.copy()
                probability = kws.predict(window)

                hit = probability >= threshold
                recent_hits.append(1 if hit else 0)
                recent_hits = recent_hits[-args.hits:]

                line = f"\rKWS probability: {probability:6.3f}"
                if args.show_db:
                    rms = float(np.sqrt(np.mean(window * window) + 1e-12))
                    dbfs = 20.0 * np.log10(max(rms, 1e-8))
                    line += f" | level: {dbfs:6.1f} dBFS"

                print(line, end="", flush=True)

                now = time.monotonic()
                if (
                    len(recent_hits) == args.hits
                    and sum(recent_hits) == args.hits
                    and now - last_detection >= args.cooldown
                ):
                    print("\n\n>>> WAKE WORD DETECTED <<<\n")
                    last_detection = now
                    recent_hits.clear()

    except KeyboardInterrupt:
        print("\n\nStopped.")
    except sd.PortAudioError as exc:
        print(
            "\nMicrophone error:\n"
            f"{exc}\n\n"
            "Try: python test_kws_mic.py --list-devices\n"
            "Then select a microphone with --device N."
        )
        raise


if __name__ == "__main__":
    main()
