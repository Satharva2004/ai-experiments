"""
Ears and mouth, both running on this Mac (no cloud service):
  listen()  records the microphone and turns speech into text with Whisper (MLX, Apple Silicon)
  say()     speaks with the built-in macOS voice
"""

import subprocess

import numpy as np
import sounddevice as sd

RATE = 16000  # Whisper expects 16 kHz mono
WHISPER = "mlx-community/whisper-large-v3-turbo"  # ~1.6GB, downloaded once on first use


def listen():
    """Press Enter to start talking, Enter again to stop. Returns the text."""
    input("\n🎙  Press Enter and speak (Ctrl+C to quit)...")
    print("   Listening... press Enter when you're done.")
    chunks = []
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                        callback=lambda data, frames, t, status: chunks.append(data.copy())):
        input()
    if not chunks:
        return ""
    audio = np.concatenate(chunks).flatten()

    import mlx_whisper  # imported here so text-only use starts fast
    text = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER, language="en")["text"].strip()
    print(f"   You said: {text}")
    return text


def say(text, voice="Samantha"):
    """Speak out loud. Other voices: run  say -v '?'"""
    print(f"🔊 {text}")
    subprocess.run(["say", "-v", voice, text])
