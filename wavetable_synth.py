import torch
from torch import nn
import numpy as np
from utils import *
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
from core import upsample


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


def wavetable_osc_fast(wavetable, freq, sr):
    freq = freq.squeeze()
    increment = freq / sr * wavetable.shape[0]
    index = torch.cumsum(increment, dim=0) - increment[0]
    index = index % wavetable.shape[0]

    # linear interpolate
    index_low = torch.floor(index.clone())
    index_high = torch.ceil(index.clone())
    alpha = index - index_low
    index_low = index_low.long()
    index_high = index_high.long()

    output = wavetable[index_low] + alpha * (wavetable[index_high % wavetable.shape[0]] - wavetable[index_low])
        
    return output


class WavetableSynth(nn.Module):
    def __init__(self,
                 wavetables=None,
                 n_wavetables=64,
                 wavetable_len=512,
                 sr=44100,
                 duration_secs=3):
        super(WavetableSynth, self).__init__()
        if wavetables is None: 
            # TODO: parameterize this
            self.wavetables = torch.nn.Parameter(torch.normal(mean=torch.zeros(n_wavetables, wavetable_len + 1),
                                                              std=torch.ones(n_wavetables, wavetable_len + 1) * 0.001))
            self.wavetables.data = torch.cat([self.wavetables[:, :-1], self.wavetables[:, 0].unsqueeze(-1)], dim=-1)
            self.wavetables.requires_grad = True

            self.attention = torch.nn.Parameter(torch.normal(mean=torch.zeros(n_wavetables, sr * duration_secs),
                                                             std=torch.ones(n_wavetables, sr * duration_secs) * 1))
        else:
            self.wavetables = wavetables
        self.sr = sr

    def forward(self, pitch, amplitude, duration_secs):
        """
            pitch: (t * sr,)       # frequency
            amplitude: (t * sr,)
        """
        self.wavetables.data = torch.cat([self.wavetables[:, :-1], self.wavetables[:, 0].unsqueeze(-1)], dim=-1)
        assert (self.wavetables[:, 0] == self.wavetables[:, -1]).all()

        attention = nn.Softmax(dim=0)(self.attention)

        output_waveform = torch.zeros(duration_secs * self.sr,) 
        waveform_lst = []
        
        for wt_idx in range(self.wavetables.shape[0]):
            wt = self.wavetables[wt_idx]
            waveform = wavetable_osc_fast(wt, pitch, self.sr)
            waveform_lst.append(waveform.squeeze())
        
        # attention per wavetable
        output_waveform = torch.stack(waveform_lst, dim=0)
        output_waveform = output_waveform * attention
        
        # sum into final output
        output_waveform = torch.sum(output_waveform, dim=0)
        output_waveform = output_waveform.unsqueeze(0).unsqueeze(-1)
        output_waveform = output_waveform * amplitude
        
        return output_waveform


if __name__ == "__main__":
    # create a sine wavetable
    wavetable_len = 512
    sr = 44100
    duration = 3
    freq = 739.99
    sine_wavetable = generate_wavetable(wavetable_len, np.sin)
    # sawtooth_wavetable = generate_wavetable(wavetable_len, sawtooth_waveform)
    wavetable = torch.tensor([sine_wavetable,])
    
    wt_synth = WavetableSynth(wavetables=wavetable, sr=sr)
    freq_t = torch.ones(sr * duration,) * freq
    amplitude_t = torch.ones(sr * duration,)

    print(freq_t, 'freq')
    y = wt_synth(freq_t, amplitude_t, duration)
    # sf.write('test_3s.wav', y, sr, 'PCM_24')
    plt.plot(y[1000:2000])
    plt.show()



