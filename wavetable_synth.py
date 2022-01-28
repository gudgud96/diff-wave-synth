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
    index = torch.cumsum(increment, dim=1) - increment[0]
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
            self.wavetables = nn.ParameterList([nn.Parameter(torch.normal(mean=torch.zeros(wavetable_len),
                                                                          std=torch.ones(wavetable_len) * 0.01)) for _ in range(n_wavetables)])

            for wt in self.wavetables:
                wt.data = torch.cat([wt[:-1], wt[0].unsqueeze(-1)], dim=-1)
                wt.requires_grad = True
            
            # self.wavetables = torch.nn.Parameter(torch.normal(mean=torch.zeros(n_wavetables, wavetable_len + 1),
            #                                                   std=torch.ones(n_wavetables, wavetable_len + 1) * 0.01))
            # self.wavetables.data = torch.cat([self.wavetables[:, :-1], self.wavetables[:, 0].unsqueeze(-1)], dim=-1)
            # self.wavetables.requires_grad = True

            # self.attention = None
            # self.attention = nn.Parameter(torch.zeros(n_wavetables, 100 * duration_secs))
            # torch.nn.init.xavier_uniform(self.attention)

        else:
            self.wavetables = wavetables
            self.attention = nn.Parameter(torch.normal(mean=torch.zeros(n_wavetables, sr * duration_secs),
                                                       std=torch.ones(n_wavetables, sr * duration_secs) * 1))
        self.sr = sr
        self.attention_mlp = nn.Conv1d(1, 1, 5, stride=1, padding=2)
        self.attention_softmax = nn.Softmax(dim=0)

    def forward(self, pitch, amplitude, y, duration_secs):
        """
            pitch: (t * sr,)       # frequency
            amplitude: (t * sr,)
        """
        y = y.cuda()
        
        # output_waveform = torch.zeros(pitch.shape[0], pitch.shape[1]).cuda()
        output_waveform = []
        attention = []
        for wt_idx in range(len(self.wavetables)):
            wt = self.wavetables[wt_idx]
            waveform = wavetable_osc_fast(wt, pitch, self.sr)

            cur_att = waveform * y
            cur_att = torch.mean(cur_att, dim=0)
            attention.append(cur_att)
            output_waveform.append(waveform)

        attention = torch.stack(attention, dim=0).unsqueeze(1)
        attention = self.attention_mlp(attention).squeeze()
        attention = self.attention_softmax(attention)

        self.attention = attention

        output_waveform = torch.stack(output_waveform, dim=1)
        output_waveform = output_waveform * attention
        output_waveform = torch.sum(output_waveform, dim=1)
      
        output_waveform = output_waveform.unsqueeze(-1)
        output_waveform = output_waveform * amplitude
       
        return output_waveform


if __name__ == "__main__":
    # create a sine wavetable
    wavetable_len = 512
    sr = 16000
    duration = 4
    freq_t = [739.99 for _ in range(sr)] + [523.25 for _ in range(sr)] + [349.23 for _ in range(sr * 2)]
    freq_t = torch.tensor(freq_t)
    freq_t = torch.stack([freq_t, freq_t, freq_t], dim=0)
    sine_wavetable = generate_wavetable(wavetable_len, np.sin)
    # sawtooth_wavetable = generate_wavetable(wavetable_len, sawtooth_waveform)
    wavetable = torch.tensor([sine_wavetable,])
    
    wt_synth = WavetableSynth(wavetables=wavetable, sr=sr, duration_secs=4)
    # freq_t = torch.ones(sr * duration,) * freq
    amplitude_t = torch.ones(sr * duration,)
    amplitude_t = torch.stack([amplitude_t, amplitude_t, amplitude_t], dim=0)
    amplitude_t = amplitude_t.unsqueeze(-1)

    # print(freq_t, 'freq')
    y = wt_synth(freq_t, amplitude_t, duration)
    print(y.shape, 'y')
    plt.plot(y.squeeze()[0].detach().numpy())
    plt.show()
    sf.write('test_3s_v1.wav', y.squeeze()[0].detach().numpy(), sr, 'PCM_24')
    sf.write('test_3s_v2.wav', y.squeeze()[1].detach().numpy(), sr, 'PCM_24')
    sf.write('test_3s_v3.wav', y.squeeze()[2].detach().numpy(), sr, 'PCM_24')
    # plt.plot(y[1000:2000])
    # plt.show()



