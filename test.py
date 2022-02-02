"""
Test script.
"""
import numpy as np
from core import multiscale_fft
import torch
import yaml 
from dataloader import get_data_loader
from tqdm import tqdm
from model import WTS
from nnAudio import Spectrogram
import soundfile as sf
import matplotlib.pyplot as plt

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
sr = config["common"]["sampling_rate"]
block_size = config["common"]["block_size"]
duration_secs = config["common"]["duration_secs"]
batch_size = config["train"]["batch_size"]
scales = config["test"]["scales"]
overlap = config["test"]["overlap"]

model = WTS(hidden_size=512, n_harmonic=100, n_bands=65, sampling_rate=sr,
            block_size=block_size, n_wavetables=10, mode="wavetable", 
            duration_secs=duration_secs)
model.cuda()
model.load_state_dict(torch.load("model.pt"))
spec = Spectrogram.MFCC(sr=sr, n_mfcc=30)

mean_loudness, std_loudness = -39.74668743704927, 54.19612404969509

test_dl = get_data_loader(config, mode="test", batch_size=batch_size)

for y, loudness, pitch in tqdm(test_dl):
    mfcc = spec(y)
    pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
    loudness = (loudness - mean_loudness) / std_loudness

    plt.plot(loudness[1].squeeze().numpy())
    plt.show()

    mfcc = mfcc.cuda()
    pitch = pitch.cuda()
    loudness = loudness.cuda()

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
        s_x = s_x.cuda()
        s_y = s_y.cuda()
        lin_loss = ((s_x - s_y).abs()).mean()
        loss += lin_loss

    print("Test Loss: {:.4}".format(loss.item()))
    print(output.shape, y.shape)

