import torch
from torch import nn
import numpy as np
from utils import *
from tqdm import tqdm
import soundfile as sf


def linear_interpolator(wavetable, index):
    low = int(index)
    high = int(np.ceil(index))

    # Handle integer index
    if low == high:
        return wavetable[low]

    # Return the weighted sum
    return (index - low) * wavetable[high % wavetable.shape[0]] + \
            (high - index) * wavetable[low]


def wavetable_osc(wavetable, freq, sr, interpolator, duration_secs):
    buffer = torch.zeros((duration_secs * sr,))
    # TODO: remove for loop it's super slow
    index = 0
    for idx in tqdm(range(len(buffer))):
        increment = wavetable.shape[0] * freq[idx] / sr
        sample = interpolator(wavetable, index)
        buffer[idx] = sample
        index = (index + increment) % wavetable.shape[0]
    return buffer


class WavetableSynth(nn.Module):
    def __init__(self,
                 wavetables=None,
                 n_wavetables=64,
                 wavetable_len=512,
                 sr=44100):
        super(WavetableSynth, self).__init__()
        if wavetables is None: 
            # TODO: parameterize this
            self.wavetables = torch.rand((n_wavetables, wavetable_len))
        else:
            self.wavetables = wavetables
        self.sr = sr

    def forward(self, pitch, amplitude, duration_secs):
        """
            pitch: (t * sr,)       # frequency
            amplitude: (t * sr,)
        """
        # TODO: extend to batch
        output_waveform = torch.zeros(duration_secs * self.sr,) 
        for wt in self.wavetables:
            waveform = wavetable_osc(wt, pitch, self.sr, linear_interpolator, duration_secs)
            # TODO: change to parameterizable attention later. now just do average poolling
            waveform *= torch.ones(waveform.shape) / self.wavetables.shape[0]
            output_waveform += waveform
        
        output_waveform *= amplitude
        return output_waveform


if __name__ == "__main__":
    # create a sine wavetable
    wavetable_len = 512
    sr = 44100
    duration = 3
    freq = 523.25
    sine_wavetable = generate_wavetable(wavetable_len, np.sin)
    sawtooth_wavetable = generate_wavetable(wavetable_len, sawtooth_waveform)
    wavetable = torch.tensor([sine_wavetable, sawtooth_wavetable])
    
    wt_synth = WavetableSynth(wavetables=wavetable, sr=sr)
    freq_t = torch.ones(sr * duration,) * freq
    amplitude_t = torch.ones(sr * duration,)

    y = wt_synth(freq_t, amplitude_t, duration)
    sf.write('test_3s.wav', y, sr, 'PCM_24')



