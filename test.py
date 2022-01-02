import librosa
import soundfile as sf
import numpy as np
from core import extract_loudness, extract_pitch, multiscale_fft
from model import WTS, DDSPv2
import torch
import matplotlib.pyplot as plt
from nnAudio import Spectrogram
import librosa.display

# filename = librosa.ex('trumpet')
# y, sr = librosa.load("trumpet.mp3", duration=2.0)
# print(y.shape)

scales = [4096, 2048, 1024, 512, 256, 128]
overlap = .75
duration = 3
model = WTS(hidden_size=512, n_harmonic=100, n_bands=65, sampling_rate=44100,
            block_size=441, n_wavetables=3, mode="wavetable", duration_secs=duration)
# model = DDSPv2(hidden_size=512, n_harmonic=100, n_bands=65, sampling_rate=44100,
#                block_size=441)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# spec = Spectrogram.MelSpectrogram(sr=44100)
spec = Spectrogram.MFCC(sr=44100, n_mfcc=20)

# sf.write('test_trumpet_3s.wav', y, sr, 'PCM_24')

def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    x, sr = librosa.load(f, sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    print(pitch.shape, loudness.shape, 'pl')

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness

# x, pitch, loudness = preprocess('test_trumpet_3s.wav', 44100, 441, 44100 * duration, True)
x, pitch, loudness = preprocess('test_flute.mp3', 44100, 441, 44100 * duration, True)
mfcc = spec(torch.tensor(x))
pitch, loudness = torch.tensor(pitch).unsqueeze(-1).float(), torch.tensor(loudness).unsqueeze(-1).float()
mean_l, std_l = torch.mean(loudness), torch.std(loudness)

for ep in range(200):
    l = (loudness - mean_l) / std_l
    output = model(mfcc, pitch, l)
    ori_stft = multiscale_fft(
                torch.tensor(x).squeeze(),
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
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (torch.log(s_x + 1e-7) - torch.log(s_y + 1e-7)).abs().mean()
        loss += lin_loss

        cum_lin_loss += lin_loss
        cum_log_loss += log_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    print("Loss {}: {:.4} recon: {:.4} {:.4}".format(ep, loss.item(), cum_lin_loss.item(), cum_log_loss.item()))

input_wav = x.squeeze()
output_wav = output.squeeze().detach().numpy()

D = librosa.amplitude_to_db(np.abs(librosa.stft(input_wav)), ref=np.max)
librosa.display.specshow(D, y_axis='log', sr=44100, hop_length=1024,
                         x_axis='time')
plt.show()

D = librosa.amplitude_to_db(np.abs(librosa.stft(output_wav)), ref=np.max)
librosa.display.specshow(D, y_axis='log', sr=44100, hop_length=1024,
                         x_axis='time')
plt.show()
for i in range(len(model.wts.wavetables)):
    plt.plot(model.wts.wavetables[i].squeeze().detach().numpy())
    plt.title("Wavetable " + str(i))
    plt.show()

att = torch.nn.Softmax(dim=0)(model.wts.attention)
for i in range(len(att)):
    plt.plot(att[i].squeeze().detach().numpy()[::400])
    plt.title("Attention " + str(i))
    plt.show()

sf.write('test_trumpet_out_v2.wav', output_wav, 44100, 'PCM_24')
sf.write('test_trumpet_in_v2.wav', input_wav, 44100, 'PCM_24')





