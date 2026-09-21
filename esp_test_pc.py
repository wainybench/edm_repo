import re
import numpy as np
import pyaudio
import librosa
import tensorflow as tf

MODEL_PATH = "micro_kws_model.tflite"
NORMALIZATION_PATH = "normalization.h"
RATE = 16000
CHUNK = 16000  #1 second of audio per inference
CONFIDENCE_THRESHOLD = 0.85


def load_normalization(header_path=NORMALIZATION_PATH):
    #Parse the MFCC_MEAN/MFCC_STD arrays out of the C header exported by Esp.py and return them as numpy arrays
    with open(header_path, "r") as f:
        text = f.read()
    mean_str = re.search(r"MFCC_MEAN\[\]\s*=\s*\{([^}]*)\}", text).group(1)
    std_str = re.search(r"MFCC_STD\[\]\s*=\s*\{([^}]*)\}", text).group(1)
    mean = np.array([float(v.strip().rstrip('f')) for v in mean_str.split(",")], dtype=np.float32)
    std = np.array([float(v.strip().rstrip('f')) for v in std_str.split(",")], dtype=np.float32)
    return mean, std


def quantize(input_data, quant_params):
    #converts float32 input data to int8 using the quantization parameters from the TFLite model
    scale, zero_point = quant_params
    q = np.round(input_data / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(output_value, quant_params):
    #Convert a quantized int8 output back to a 0.0-1.0 confidence
    scale, zero_point = quant_params
    return (float(output_value) - float(zero_point)) * float(scale)


def main():
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_is_int8 = input_details[0]['dtype'] == np.int8
    output_is_int8 = output_details[0]['dtype'] == np.int8

    print("Loading normalization stats...")
    mfcc_mean, mfcc_std = load_normalization()

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    print("\nListening for the keyword... (Press Ctrl+C to stop)")

    try:
        while True:
            #Read 1 full second of audio from the mic
            data = stream.read(CHUNK, exception_on_overflow=False)

            #Convert raw audio bytes to float32 in [-1.0, 1.0]
            signal = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            #Extract MFCCs
            mfcc = librosa.feature.mfcc(y=signal, sr=RATE, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc = mfcc.T

            #noormalizing with the stats from training
            mfcc = (mfcc - mfcc_mean) / mfcc_std

            #Reshape for the network: (1, time_steps, n_mfcc, 1)
            input_data = mfcc[np.newaxis, ..., np.newaxis]

            #Quantizing to int8
            if input_is_int8:
                input_data = quantize(input_data, input_details[0]['quantization'])
            else:
                input_data = input_data.astype(np.float32)

            #Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            #Dequantize the output back to a readable confidence
            if output_is_int8:
                confidence = dequantize(output_data[0][0], output_details[0]['quantization'])
            else:
                confidence = float(output_data[0][0])

            #Report
            if confidence > CONFIDENCE_THRESHOLD:
                print(f"\nKEYWORD DETECTED! (Confidence: {confidence:.2f})")
            else:
                print(f"Background noise (Confidence: {confidence:.2f})", end='\r')

    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()