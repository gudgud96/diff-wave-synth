from sre_constants import SRE_FLAG_DOTALL
import numpy as np
from core import extract_loudness, extract_pitch_v2, multiscale_fft
import torch
import yaml 
from dataloader import get_data_loader
from tqdm import tqdm
from model import WTS
from nnAudio import Spectrogram
from tensorboardX import SummaryWriter
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
scales = config["test"]["scales"]
overlap = config["test"]["overlap"]

model = WTS(hidden_size=512, n_harmonic=100, n_bands=65, sampling_rate=sr,
            block_size=block_size, n_wavetables=10, mode="wavetable", 
            duration_secs=duration_secs)
model.cuda()
model.load_state_dict(torch.load("model.pt"))
spec = Spectrogram.MFCC(sr=sr, n_mfcc=30)

mean_loudness, std_loudness = -39.74668743704927, 54.19612404969509

# test_dl = get_data_loader(config, mode="train", batch_size=batch_size)

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
    cum_lin_loss = 0
    cum_log_loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        s_x = s_x.cuda()
        s_y = s_y.cuda()
        
        lin_loss = ((s_x - s_y).abs()).mean()
        # lin_loss = torch.sum(lin_loss, dim=(1, 2)).mean()
        print(lin_loss, 'lin_loss')
        log_loss = (torch.log(s_x + 1e-7) - torch.log(s_y + 1e-7)).abs().mean()
        loss += lin_loss

        cum_lin_loss += lin_loss
        cum_log_loss += log_loss

    print("Loss: {:.4}".format(loss.item()))
    print(output.shape, y.shape)
    break


fig, axs = plt.subplots(2)
axs[0].plot(output.squeeze().cpu().detach().numpy()[1])
axs[1].plot(y.squeeze().cpu().detach().numpy()[1])
plt.show()


for idx in range(16):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y.squeeze().cpu().detach().numpy()[idx])), ref=np.max)
    librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=512,
                            x_axis='time')
    plt.show()

    D = librosa.amplitude_to_db(np.abs(librosa.stft(output.squeeze().cpu().detach().numpy()[idx])), ref=np.max)
    librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=512,
                            x_axis='time')
    plt.show()

    sf.write('test_model_out_v1_{}.wav'.format(idx), output.squeeze().cpu().detach().numpy()[idx], sr, 'PCM_24')
    sf.write('test_model_in_v1_{}.wav'.format(idx), y.squeeze().cpu().detach().numpy()[idx], sr, 'PCM_24')

