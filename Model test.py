import pyaudio
import json
from vosk import Model , KaldiRecognizer

#Loading the sihmod
print("model loading")
try:
    sihmod = Model("sihmod")
except Exception as e:
    print("Error Thrown, model no found")
    exit()

#sampling rate set to 16k hz
recognizer = KaldiRecognizer(sihmod, 16000)

#pyaudio setup for input from microphone(my lappys mic)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, 
                channels=1, 
                rate=16000, 
                input=True, 
                frames_per_buffer=8000)
stream.start_stream()

print("Model loaded, its now listening")

#loop for listenign
try:
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            print(f"\nFinal Transcribed Command {result['text']} <<<")
        else:
            partial_result = json.loads(recognizer.PartialResult())
            print(f"Listening: {partial_result['partial']}", end='\r')
except KeyboardInterrupt:
    print("\nStopping ASR test...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()