from core import harmonic_synth
from wavetable_synth import WavetableSynth
import torch
import torch.nn as nn
from core import mlp, gru, scale_function, remove_above_nyquist, upsample
from core import amp_to_impulse_response, fft_convolve
from core import resample
import math
from torchvision.transforms import Resize


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class WTS(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size, n_wavetables, mode="wavetable", duration_secs=3):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = mlp(20, hidden_size, 3)
        self.layer_norm = nn.LayerNorm(20)
        self.gru_mfcc = nn.GRU(20, 512, batch_first=True)
        self.mlp_mfcc = nn.Linear(512, 16)

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3),
                                      mlp(1, hidden_size, 3),
                                      mlp(16, hidden_size, 3)])
        self.gru = gru(3, hidden_size)
        self.out_mlp = mlp(hidden_size * 4, hidden_size, 3)

        self.loudness_mlp = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)
        self.wts = WavetableSynth(n_wavetables=n_wavetables, sr=sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.mode = mode
        self.duration_secs = duration_secs

    def forward(self, mfcc, pitch, loudness):
        # encode mfcc first
        # use layer norm instead of trainable norm
        mfcc = self.layer_norm(torch.transpose(mfcc, 1, 2))
        mfcc = self.gru_mfcc(mfcc)[0]
        mfcc = self.mlp_mfcc(mfcc)

        # use image resize, ddsp also do this so...haha
        mfcc = Resize(size=(self.duration_secs * 100, 16))(mfcc)

        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
            self.in_mlps[2](mfcc)
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], hidden], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = self.proj_matrices[0](hidden)
        if self.mode != "wavetable":
            param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        total_amp_2 = self.loudness_mlp(loudness)

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)
        total_amp = upsample(total_amp, self.block_size)    # this one can't use, found problems when back propagated, weird
        total_amp_2 = upsample(total_amp_2, self.block_size)    # use this instead for wavetable

        # replace with wavetable synthesis
        if self.mode == "wavetable":
            print(self.mode)
            harmonic = self.wts(pitch, total_amp_2, self.duration_secs)
            # harmonic = harmonic.unsqueeze(0).unsqueeze(-1)  # TODO: this should be remove too
        else:
            print(self.mode)
            harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal


class DDSPv2(nn.Module):
    """
    with encoder, input is mfcc
    """
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = mlp(20, hidden_size, 3)
        self.layer_norm = nn.LayerNorm(20)
        self.gru_mfcc = nn.GRU(20, 512, batch_first=True)
        self.mlp_mfcc = nn.Linear(512, 16)

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3),
                                      mlp(1, hidden_size, 3),
                                      mlp(16, hidden_size, 3)])
        self.gru = gru(3, hidden_size)
        self.out_mlp = mlp(hidden_size * 4, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, mfcc, pitch, loudness):
        # encode mfcc first
        # use layer norm instead of trainable norm
        mfcc = self.layer_norm(torch.transpose(mfcc, 1, 2))
        mfcc = self.gru_mfcc(mfcc)[0]
        mfcc = self.mlp_mfcc(mfcc)

        # use image resize, ddsp also do this so...haha
        mfcc = Resize(size=(300, 16))(mfcc)

        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
            self.in_mlps[2](mfcc)
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], hidden], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal