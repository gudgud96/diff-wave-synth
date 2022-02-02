"""
Train script.
"""
import numpy as np
from core import multiscale_fft, get_scheduler
import torch
import yaml 
from dataloader import get_data_loader
from tqdm import tqdm
from model import WTS
from nnAudio import Spectrogram
import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

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
epochs = config["train"]["epochs"]

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

model = WTS(hidden_size=hidden_size, n_harmonic=n_harmonic, n_bands=n_bands, sampling_rate=sr,
            block_size=block_size, n_wavetables=n_wavetables, mode="wavetable", 
            duration_secs=duration_secs)
model.cuda()
opt = torch.optim.Adam(model.parameters(), lr=train_lr)
spec = Spectrogram.MFCC(sr=sr, n_mfcc=n_mfcc)

# both values are pre-computed from the train set 
mean_loudness, std_loudness = -39.74668743704927, 54.19612404969509

train_dl = get_data_loader(config, mode="train", batch_size=batch_size)
valid_dl = get_data_loader(config, mode="valid", batch_size=batch_size)

# for now the scheduler is not used
schedule = get_scheduler(
    len(train_dl),
    config["train"]["start_lr"],
    config["train"]["stop_lr"],
    config["train"]["decay_over"],
)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/' + current_time +'/train'

train_summary_writer = SummaryWriter(train_log_dir)

idx = 0
for ep in tqdm(range(1, epochs + 1)):
    for y, loudness, pitch in tqdm(train_dl):
        mfcc = spec(y)
        pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
        loudness = (loudness - mean_loudness) / std_loudness

        mfcc = mfcc.cuda()
        pitch = pitch.cuda()
        loudness = loudness.cuda()

        output = model(mfcc, pitch, loudness)
        
        ori_stft = multiscale_fft(
                    torch.tensor(y).squeeze(),
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
            
            lin_loss = (s_x - s_y).abs().mean()
            loss += lin_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_summary_writer.add_scalar('loss', loss.item(), global_step=idx)
        if idx % 500 == 0:
            torch.save(model.state_dict(), "model.pt")
        
        idx += 1
        