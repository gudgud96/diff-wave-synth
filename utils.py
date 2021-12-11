import numpy as np


def generate_wavetable(length, f):
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(2 * np.pi * i / length)
    return wavetable


def sawtooth_waveform(x):
    """Sawtooth with period 2 pi."""
    return (x + np.pi) / np.pi % 2 - 1


def square_waveform(x):
    """Square waveform with period 2 pi."""
    return np.sign(np.sin(x))