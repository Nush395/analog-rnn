import numpy as np
import librosa
import os


def normalize_waveform(wav_data):
    """Normalize the amplitude of a vowel waveform

    Args:
        wav_data: Numpy array of bits representing audio file
    """
    total_power = np.square(wav_data).sum()

    return wav_data / np.sqrt(total_power)


def load_audio(file, sr=None, normalize=True):
    """Use librosa to to load a single audio file with a specified sample rate

    Args:
        file: Handle to a file
        sr: sampling rate of the file
        normalize: Boolean of whether to normalize audio or not
    """

    data, rate = librosa.load(file, sr=sr)

    if normalize:
        data = normalize_waveform(data)

    return data


def load_all_audio(dir, sr=16000):
    """Loads all audio files in a given directory

    Args:
        dir: Path to directory with wav files
        sr: sampling rate (default 16k)
    """
    data = []
    labels = []
    for fil in os.listdir(dir):
        with open(fil) as f:
            d = load_audio(f, sr)
            data.append(d)
        label = os.path.basename(fil).split('.')[0]
        labels.append(label)
    return data, labels
