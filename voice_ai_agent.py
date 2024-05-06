import time

from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self):
        pass
    
    def load_model(self):
        stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        return stt_model

    def run(self, model, audio):
        segments, _ = model.trascribe(audio)
        return segments
        

if __name__ == "__main__":
    audio = "path/to/wav/file.wav"
    stt_obj = SpeechToText()
    st = time.time()
    stt_model = stt_obj.load_model()
    print(f"Time taken to load model: {(time.time() - st) * 1000} ms")
    st = time.time()
    stt_model.run(stt_model, audio)
    print(f"Inference Model: {(time.time() - st) * 1000} ms")
