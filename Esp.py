import os
import numpy as np
import librosa
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

layers = keras.layers
models = keras.models

#configuring the whole dataset
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000
DURATION = 1.0  # 1 second audio clips
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 13


def extract_mfcc(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    #isolating the voice logs
    signal, _ = librosa.effects.trim(signal, top_db=30)

    if len(signal) > SAMPLES_PER_TRACK:
        start = (len(signal) - SAMPLES_PER_TRACK) // 2
        signal = signal[start:start + SAMPLES_PER_TRACK]
    else:
        pad_total = SAMPLES_PER_TRACK - len(signal)
        pad_left = pad_total // 2
        signal = np.pad(signal, (pad_left, pad_total - pad_left), "constant")

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    return mfcc.T


def load_data():
    #Extracting MFCC feats from the dataset and returning them as arrays X (features) and y (labels)
    X, y = [], []
    labels = ["noise", "keyword"]

    for i, label in enumerate(labels):
        folder_path = os.path.join(DATASET_PATH, label)
        print(f"Folder: {label}...")
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                X.append(extract_mfcc(file_path))
                y.append(i)  # 0 = noise, 1 = keyword

    return np.array(X), np.array(y)


def export_c_array(data_bytes, array_name, header_path):
    #Writes raw byte data to a C header file as a uint8_t array
    hex_array = ", ".join(f"0x{b:02x}" for b in data_bytes)
    with open(header_path, "w") as f:
        f.write("#include <cstdint>\n\n")
        f.write(f"alignas(8) const unsigned char {array_name}[] = {{\n")
        f.write(hex_array)
        f.write("\n};\n")
        f.write(f"const int {array_name}_len = {len(data_bytes)};\n")


def export_normalization_header(mean, std, header_path="normalization.h"):
    #writes the noormalizations s oesp32 can have a refrence
    mean_flat = mean.flatten()
    std_flat = std.flatten()
    with open(header_path, "w") as f:
        f.write("// Auto-generated MFCC normalization constants\n")
        f.write("// Apply on-device as: (mfcc[i] - MFCC_MEAN[i]) / MFCC_STD[i]\n\n")
        f.write(f"const int MFCC_NUM_COEFFS = {len(mean_flat)};\n\n")
        f.write("const float MFCC_MEAN[] = {" + ", ".join(f"{v:.6f}f" for v in mean_flat) + "};\n")
        f.write("const float MFCC_STD[] = {" + ", ".join(f"{v:.6f}f" for v in std_flat) + "};\n")


#loading data
print("Loading data and extracting features...")
X, y = load_data()  # X: (N, time_steps, n_mfcc)

#splitting before normalization avaoiding leaks
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#training set statistics for normalization
mfcc_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
mfcc_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train_raw - mfcc_mean) / mfcc_std
X_test = (X_test_raw - mfcc_mean) / mfcc_std

#channel dim for coonv2d (N, time_steps, n_mfcc, 1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

input_shape = (X_train.shape[1], X_train.shape[2], 1)

#tiny cnn model
print("Building the TinyML Model...")
model = models.Sequential([
    layers.SeparableConv2D(8, (3, 3), activation='relu', input_shape=input_shape),
    #sepconv is for dscnn, it is less accurate
    #layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.SeparableConv2D(16, (3, 3), activation='relu'),
    #sepconv is for dscnn, it is less accurate
    #layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),  # 0 = noise, 1 = keyword
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(name = 'Precision'), keras.metrics.Recall(name = 'Recall')])

model.summary()

#training
print("Training started")
model.fit(X_train, y_train, epochs = 35, batch_size=16, validation_data=(X_test, y_test))


probs = model.predict(X_test).ravel()
print(classification_report(y_test, probs > 0.85, target_names=["noise", "keyword"]))

p, r, thr = precision_recall_curve(y_test, probs)
#quantizing for esp
print("Quantizing model to INT8 for ESP32")


def representative_dataset():
    num_samples = min(100, len(X_train))
    for i in range(num_samples):
        yield [X_train[i].astype(np.float32)[np.newaxis, ...]]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

def predict_int8(X):
    interp = tf.lite.Interpreter(model_content=tflite_quant_model)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    (i_s, i_z), (o_s, o_z) = inp["quantization"], out["quantization"]
    probs = []
    for x in X:
        q = np.clip(np.round(x / i_s + i_z), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[np.newaxis, ...])
        interp.invoke()
        probs.append((interp.get_tensor(out["index"])[0][0] - o_z) * o_s)
    return np.array(probs)

q_probs = predict_int8(X_test)
print(classification_report(y_test, q_probs > 0.85, target_names=["noise", "keyword"]))

with open("micro_kws_model.tflite", "wb") as f:
    f.write(tflite_quant_model)

#in and out quant reports/loogs
interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
in_scale, in_zero = input_details["quantization"]
out_scale, out_zero = output_details["quantization"]
print(f"\nInput  quantization -> scale: {in_scale}, zero_point: {in_zero}")
print(f"Output quantization -> scale: {out_scale}, zero_point: {out_zero}")

#exporting model and norm stats for esp32
print("Converting TFLite model and normalization stats to C headers...")
export_c_array(tflite_quant_model, "model_tflite", "model.h")
export_normalization_header(mfcc_mean, mfcc_std, "normalization.h")
print("Done! Include 'model.h' and 'normalization.h' in your ESP32 project.")
