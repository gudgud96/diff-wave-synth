"""
Preprocess NSynth dataset to extract pitch and loudness.
"""
import librosa
import numpy as np
from core import extract_loudness, extract_pitch, extract_pitch_v2
from multiprocessing.dummy import Pool as ThreadPool
import yaml 
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

# if you find problems using TF, disable the GPU and inspect
# tf.config.set_visible_devices([], 'GPU')


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, 
               target_pitch_file, target_loudness_file, pitch_model_capacity="full", **kwargs):
    if os.path.exists(os.path.join(target_pitch_file, f.split("/")[-1].replace(".wav", "_pitch.npy"))):
        print("Skipping...")
    
    else:
        x, sr = librosa.load(f, sampling_rate)
        N = (signal_length - len(x) % signal_length) % signal_length
        x = np.pad(x, (0, N))

        if oneshot:
            x = x[..., :signal_length]

        pitch = extract_pitch(x, sampling_rate, block_size, model_capacity=pitch_model_capacity)
        # v2 is based on my own version of torchcrepe, comment out for now
        # pitch = extract_pitch_v2(x, sampling_rate, block_size)
        loudness = extract_loudness(x, sampling_rate, block_size)

        x = x.reshape(-1, signal_length)
        pitch = pitch.reshape(x.shape[0], -1).squeeze()
        loudness = loudness.reshape(x.shape[0], -1)

        np.save(os.path.join(target_pitch_file, f.split("/")[-1].replace(".wav", "_pitch.npy")), pitch.squeeze())
        np.save(os.path.join(target_loudness_file, f.split("/")[-1].replace(".wav", "_loudness.npy")), loudness.squeeze())

        return x, pitch, loudness


if __name__ == "__main__":
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    asyncc = True
    paths = ["train_path", "valid_path", "test_path"]
    
    for path in paths:
        audio_path_prefix = os.path.join(config["dataset"][path], config["dataset"]["audio"])
        audio_path = sorted(os.listdir(audio_path_prefix))

        print('Length: ', len(audio_path))

        if not os.path.exists(os.path.join(config["dataset"][path], config["dataset"]["pitch"])):
            os.mkdir(os.path.join(config["dataset"][path], config["dataset"]["pitch"]))
            os.mkdir(os.path.join(config["dataset"][path], config["dataset"]["loudness"]))

        if asyncc:
            pool = ThreadPool(4)
            pbar = tqdm(total=len(audio_path))

            def update(*a):
                pbar.update()

            for i in range(pbar.total):
                pool.apply_async(preprocess, 
                                args=(os.path.join(audio_path_prefix, audio_path[i]),
                                    config["common"]["sampling_rate"], 
                                    config["common"]["block_size"], 
                                    config["common"]["sampling_rate"] * config["common"]["duration_secs"], 
                                    True,
                                    os.path.join(config["dataset"][path], config["dataset"]["pitch"]),
                                    os.path.join(config["dataset"][path], config["dataset"]["loudness"])), 
                                callback=update)
            pool.close()
            pool.join()
        
        else:
            for i in tqdm(range(len(audio_path))):
                preprocess(os.path.join(audio_path_prefix, audio_path[i]), 
                            config["common"]["sampling_rate"], 
                            config["common"]["block_size"], 
                            config["common"]["sampling_rate"] * config["common"]["duration_secs"], 
                            True,
                            os.path.join(config["dataset"][path], config["dataset"]["pitch"]),
                            os.path.join(config["dataset"][path], config["dataset"]["loudness"]),
                            pitch_model_capacity=config["crepe"]["model"])