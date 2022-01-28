import numpy as np
from core import multiscale_fft, get_scheduler
import torch
import yaml 
from dataloader import get_data_loader
from tqdm import tqdm
from model import WTS
from nnAudio import Spectrogram
import time
import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from torchviz import make_dot

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# general parameters
sr = config["common"]["sampling_rate"]
block_size = config["common"]["block_size"]
duration_secs = config["common"]["duration_secs"]
batch_size = config["train"]["batch_size"]
scales = config["train"]["scales"]
overlap = config["train"]["overlap"]

model = WTS(hidden_size=512, n_harmonic=100, n_bands=65, sampling_rate=sr,
            block_size=block_size, n_wavetables=10, mode="wavetable", 
            duration_secs=duration_secs)
model.cuda()
# model.load_state_dict(torch.load("model_wts_v1.pt"))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
spec = Spectrogram.MFCC(sr=sr, n_mfcc=30)

mean_loudness, std_loudness = -39.74668743704927, 54.19612404969509

train_dl = get_data_loader(config, mode="train", batch_size=batch_size)
valid_dl = get_data_loader(config, mode="valid", batch_size=batch_size)

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
for ep in tqdm(range(1, 100001)):
    for y, loudness, pitch in tqdm(train_dl):
        mfcc = spec(y)
        pitch, loudness = pitch.unsqueeze(-1).float(), loudness.unsqueeze(-1).float()
        # mean_l, std_l = torch.mean(loudness), torch.std(loudness)   # TODO: change this

        loudness = (loudness - mean_loudness) / std_loudness

        mfcc = mfcc.cuda()
        pitch = pitch.cuda()
        loudness = loudness.cuda()

        output = model(y, mfcc, pitch, loudness)
        
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
        cum_lin_loss = 0
        cum_log_loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            s_x = s_x.cuda()
            s_y = s_y.cuda()
            
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (torch.log(s_x + 1e-7) - torch.log(s_y + 1e-7)).abs().mean()
            loss += lin_loss

            cum_lin_loss += lin_loss
            cum_log_loss += log_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        # print(model.wts.wavetables.grad[0][:20], "0")
        # print(model.wts.wavetables.grad[1][:20], "0")
        # print(model.wts.wavetables.grad[2][:20], "0")

        # print("Loss {}: {:.4} recon: {:.4} {:.4}".format(ep, loss.item(), cum_lin_loss.item(), cum_log_loss.item()))
        train_summary_writer.add_scalar('loss', loss.item(), global_step=idx)
        if idx % 500 == 0:
            torch.save(model.state_dict(), "model.pt")
            # for k in range(512):
            #     train_summary_writer.add_scalar('wt_1', model.wts.wavetables[0].data.cpu().detach().numpy()[k], k)
        
        if idx % 100 == 0:
            fig, axs = plt.subplots(3, 3)
            axs[0, 0].plot(model.wts.wavetables[0].data.cpu().detach().numpy())
            axs[0, 1].plot(model.wts.wavetables[1].data.cpu().detach().numpy())
            axs[0, 2].plot(model.wts.wavetables[2].data.cpu().detach().numpy())
            axs[1, 0].plot(model.wts.wavetables[3].data.cpu().detach().numpy())
            axs[1, 1].plot(model.wts.wavetables[4].data.cpu().detach().numpy())
            axs[1, 2].plot(model.wts.wavetables[5].data.cpu().detach().numpy())
            axs[2, 0].plot(model.wts.wavetables[6].data.cpu().detach().numpy())
            axs[2, 1].plot(model.wts.wavetables[7].data.cpu().detach().numpy())
            axs[2, 2].plot(model.wts.wavetables[8].data.cpu().detach().numpy())
            plt.show()

            fig, axs = plt.subplots(3, 3)
            axs[0, 0].plot(model.wts.attention[0].data.cpu().detach().numpy()[100:150])
            axs[0, 1].plot(model.wts.attention[1].data.cpu().detach().numpy()[100:150])
            axs[0, 2].plot(model.wts.attention[2].data.cpu().detach().numpy()[100:150])
            axs[1, 0].plot(model.wts.attention[3].data.cpu().detach().numpy()[100:150])
            axs[1, 1].plot(model.wts.attention[4].data.cpu().detach().numpy()[100:150])
            axs[1, 2].plot(model.wts.attention[5].data.cpu().detach().numpy()[100:150])
            axs[2, 0].plot(model.wts.attention[6].data.cpu().detach().numpy()[100:150])
            axs[2, 1].plot(model.wts.attention[7].data.cpu().detach().numpy()[100:150])
            axs[2, 2].plot(model.wts.attention[8].data.cpu().detach().numpy()[100:150])
            plt.title("Attention")
            plt.show()

            # print(torch.mean(model.wts.wavetables[0].grad))
            # print(torch.mean(model.wts.attention.grad))
        idx += 1
        