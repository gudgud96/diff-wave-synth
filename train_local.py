"""
Simple test code with single example to make sure things work.
"""
import numpy as np
from core import extract_loudness, extract_pitch, multiscale_fft
import torch
import yaml 
from tqdm import tqdm
from model import WTS
from nnAudio import Spectrogram, features
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display


with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
sr = config["common"]["sampling_rate"]
block_size = config["common"]["block_size"]
duration_secs = config["common"]["duration_secs"]
batch_size = config["train"]["batch_size"]
scales = config["train"]["scales"]
overlap = config["train"]["overlap"]
hidden_size = config["train"]["hidden_size"]
n_harmonic = config["train"]["n_harmonic"]
n_bands = config["train"]["n_bands"]
n_wavetables = config["train"]["n_wavetables"]
n_mfcc = config["train"]["n_mfcc"]
train_lr = config["train"]["start_lr"]
visualize = config["visualize"]

print("""
======================
sr: {}
block_size: {}
duration_secs: {}
batch_size: {}
scales: {}
overlap: {}
hidden_size: {}
n_harmonic: {}
n_bands: {}
n_wavetables: {}
n_mfcc: {}
train_lr: {}
======================
""".format(sr, block_size, duration_secs, batch_size, scales, overlap,
           hidden_size, n_harmonic, n_bands, n_wavetables, n_mfcc, train_lr))

# initialize model
model = WTS(hidden_size=hidden_size, n_harmonic=n_harmonic, n_bands=n_bands, sampling_rate=sr,
            block_size=block_size, n_wavetables=n_wavetables, mode="wavetable", 
            duration_secs=duration_secs)
spec = Spectrogram.MFCC(sr=sr, n_mfcc=n_mfcc)
opt = torch.optim.Adam(model.parameters(), lr=train_lr)
mean_loudness, std_loudness = -39.74668743704927, 54.19612404969509
spec_layer = features.STFT(n_fft=2048, freq_bins=None, hop_length=512,
                           window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                           output_format='Magnitude',
                           fmin=50,fmax=11025, sr=sr)


def preprocess(f, sampling_rate, block_size, signal_length, oneshot=True):
    x, sr = librosa.load(f, sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1).squeeze()
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness

# sample downloaded from Differentiable Wavetable Synthesis online supplement
y, pitch, loudness = preprocess("test_audio/gt_2.wav", sr, block_size, sr * duration_secs)
y = torch.tensor(y)
pitch = torch.tensor(pitch).unsqueeze(0)
loudness = torch.tensor(loudness)

y = torch.cat([y, y], dim=0)
pitch = torch.cat([pitch, pitch], dim=0)
loudness = torch.cat([loudness, loudness], dim=0)

pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
loudness = (loudness - mean_loudness) / std_loudness

for ep in range(5000):
    mfcc = spec(y)
    output = model(mfcc, pitch, loudness)
    
    ori_stft = multiscale_fft(
                y.squeeze(),
                scales,
                overlap,
            )
    rec_stft = multiscale_fft(
        output.squeeze(),
        scales,
        overlap,
    )

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):        
        lin_loss = ((s_x - s_y).abs()).mean()
        loss += lin_loss

    print("{} Loss: {:.4}".format(ep, loss.item()))

    if visualize:
        spec_ori = spec_layer(y.squeeze()[0].unsqueeze(0)).cpu().detach().numpy().squeeze()
        spec_output = spec_layer(output.squeeze()[0].unsqueeze(0)).cpu().detach().numpy().squeeze()

        D_ori = librosa.amplitude_to_db(spec_ori, ref=np.max)
        D_output = librosa.amplitude_to_db(spec_output, ref=np.max)
        
        librosa.display.specshow(D_ori, y_axis='log', sr=16000, hop_length=512,
                                x_axis='time')
        plt.title("original")
        plt.show()
        librosa.display.specshow(D_output, y_axis='log', sr=16000, hop_length=512,
                                x_axis='time')
        plt.title("output")
        plt.show()
        break

    opt.zero_grad()
    loss.backward()
    opt.step()

    if ep % 200 == 0:
        torch.save(model.state_dict(), "model_test_local.pt")
        sf.write('test_ep{}.wav'.format(ep), output.squeeze().cpu().detach().numpy()[0], sr, 'PCM_24')
