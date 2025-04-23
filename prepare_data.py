import argparse
import csv
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
import torchvision
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def seed_everything(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seed set to {seed}")


def split_audio(
    audio_path,
    output_dir,
    duration=1.36,
    overlap=0.5,
    new_sample_rate=16000,
):
    """
    Split the audio file into smaller segments

    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save the smaller chunks
        duration (int): Duration of each chunk in seconds
        overlap (float): Overlap between chunks
        new_sample_rate (int): New sample rate to resample the audio to

    Returns:
        None
    """
    audio_path = Path(audio_path)
    audio_path_stem = audio_path.stem

    audio_data, sample_rate = torchaudio.load(audio_path)
    resampled_audio = F.resample(
        audio_data, orig_freq=sample_rate, new_freq=new_sample_rate
    )
    segment_samples = int(new_sample_rate * duration)
    overlap_samples = int(new_sample_rate * overlap)
    stride = segment_samples - overlap_samples

    num_segments = (resampled_audio.shape[1] - segment_samples) // stride + 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_segments):
        if i == num_segments - 1 or i == 0:
            continue
        start = i * stride
        end = start + segment_samples
        segment = resampled_audio[:1, start:end]
        output_file = os.path.join(output_dir, f"{audio_path_stem}_{i}.wav")
        torchaudio.save(output_file, segment, new_sample_rate)


def prepare_data(raw_audio_path, noise_dir, output_dir):
    # Get all the audio files
    raw_audio_files = list(Path(raw_audio_path).rglob("*.wav"))
    logging.info(f"Found {len(raw_audio_files)} audio files")

    # Delete the output directory if it exists
    if os.path.exists(output_dir):
        logging.info("Deleting existing output directory")
        shutil.rmtree(output_dir)

    # Split the audio files
    pbar = tqdm(raw_audio_files, desc="Splitting audio files")
    for audio_file in pbar:
        split_audio(
            audio_file,
            output_dir,
            duration=1.36,
            overlap=0.5,
            new_sample_rate=48000,
        )

    # train test split
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    test_ratio = 0.3
    audio_files = list(Path(output_dir).rglob("*.wav"))
    for audio_file in audio_files:
        if random.random() < test_ratio:
            shutil.move(audio_file, test_dir)
        else:
            shutil.move(audio_file, train_dir)

    # copy the noise directory
    noise_files = list(Path(noise_dir).glob("**/*.wav"))
    noise_dir = Path(output_dir) / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    noise_csv = "noise.csv"
    with open(noise_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        for noise_file in noise_files:
            writer.writerow([str(noise_file)])
            shutil.copy(noise_file, Path(output_dir) / "noise")


if __name__ == "__main__":
    seed_everything(42)

    raw_audio_path = "/home/thecw/Datasets/SSVA"
    noise_dir = "/home/thecw/Datasets/noise/"
    output_dir = "./data/ssva/"
    prepare_data(
        raw_audio_path=raw_audio_path, noise_dir=noise_dir, output_dir=output_dir
    )
