import os
import io
import whisper
import numpy as np
import pywav
import pyaudio
import wave
import speech_recognition as sr

import glob
import struct
import time

from faster_whisper import WhisperModel


def faster_whisper(recordings, model):
    faster_model = WhisperModel(model) #, device="cuda", compute_type="float16")
    for audio in recordings:
      print(f"Audio file: {audio}")
      st = time.time()
      segments, info = faster_model.transcribe(audio, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
      print(f"Time taken by faster whisper {model}: {(time.time() - st) * 1000} ms")
      print(segments)
      for segment in segments:
         print("faster whisper large v2 [%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


def record_audio(file_name, duration=2, channels=1, rate=8000, chunk=1024):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording...")

    frames = []

    # Record audio
    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # Save the recorded audio to a WAV file
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return frames


def decompress_ulaw(rawdata):
    decompressed = b"RIFF"
    datalength = len(rawdata)
    audioformat = 7
    numofchannels = 1
    sampleraate = 8000
    bitspersample = 8

    decompressed += struct.pack(
        "<L4s4sLHHLLHHH4sLL4s",
        50 + datalength,
        b"WAVE",
        b"fmt ",
        18,
        audioformat,
        numofchannels,
        sampleraate,
        int(numofchannels * sampleraate * (bitspersample / 8)),
        int(numofchannels * (bitspersample / 8)),
        bitspersample,
        0,
        b"fact",
        4,
        datalength,
        b"data",
    )
    decompressed += struct.pack("<L", datalength)
    decompressed += rawdata

    return decompressed

import audioop
def compress_ulaw(audio):
    # resamplen from 24000 to 8000
    # audio_resampled = resample(audio, orig_sr=24000, target_sr=8000)
    # quantize to PCM 16-bit
    pcm_array = np.int16(audio * 32768).tobytes()
    # convert PCM 16-bit linear to mulaw 8-bit
    mulaw_array = audioop.lin2ulaw(pcm_array, 2)
    return mulaw_array

if __name__ == "__main__":
    file_name = "recording_t.wav"
    frames = record_audio(file_name)
    # binary_bytes = b''.join(frames)
    # npm = np.frombuffer(binary_bytes, dtype=np.uint8)
    # compressed = compress_ulaw(npm)
    # decompressed = decompress_ulaw(compressed)
    # file_object = io.BytesIO(binary_bytes)
    # file_object.seek(0)
    print("Recognizing the text...")
    faster_whisper([file_name], 'large-v2')