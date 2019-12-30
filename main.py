import numpy as np
import librosa
import os


def normalize_waveform(wav_data):
    """Normalize the amplitude of a vowel waveform
    """
    total_power = np.square(wav_data).sum()

    return wav_data / np.sqrt(total_power)


def load_audio(file, sr=None, normalize=True):
    """Use librosa to to load a single audio file with a specified sample rate
    """

    data, rate = librosa.load(file, sr=sr)

    if normalize:
        data = normalize_waveform(data)

    return data

def load_all_audio(dir, sr=16000):
    """Loads all audio files in a given directory"""
    for fil in os.listdir(dir):
        with open(fil) as f:
            load_audio(f, sr)