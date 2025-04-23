import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from src.utils import seed_everything


def get_noise_waveform(
    noise_path: str | Path, duration: float
) -> tuple[torch.Tensor, int]:
    """
    Get a segment of noise waveform

    Args:
        noise_path (str | Path): path to noise file
        skip_duration (float): duration to skip in seconds
        duration (float): duration of the segment in seconds

    Returns:
        tuple[torch.Tensor, int]: noise waveform with shape (1, sample_rate * duration) and sample_rate
    """
    # Load noise waveform
    noise_path = Path(noise_path)
    noise_waveform, sample_rate = torchaudio.load(noise_path)
    audio_max_duration = noise_waveform.shape[1] / sample_rate

    # Skip some duration
    skip_duration = random.uniform(0, audio_max_duration - duration)
    start = int(skip_duration * sample_rate)
    end = start + int(duration * sample_rate)

    # Check if the duration is too long
    if end > noise_waveform.shape[1]:
        raise ValueError(
            f"Duration {duration} is too long for noise file {noise_path}. Max duration is {audio_max_duration}"
        )
    noise_waveform = noise_waveform[:1, start:end]
    return noise_waveform, sample_rate


def add_noise(
    audio_file,
    noise_file,
    snr,
):
    # Load audio data
    audio_data, sample_rate = torchaudio.load(audio_file)
    duration = audio_data.shape[1] / sample_rate

    # Load noise waveform
    noise_waveform, noise_sample_rate = get_noise_waveform(noise_file, duration)

    # Resample noise waveform and add noise to audio data
    resampled_noise_waveform = F.resample(
        noise_waveform, orig_freq=noise_sample_rate, new_freq=sample_rate
    )
    noisy_audio_data = F.add_noise(
        audio_data, resampled_noise_waveform, snr=torch.tensor([snr])
    )
    return noisy_audio_data, sample_rate


class Wav2Mel:
    def __init__(
        self,
        sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=128,
        power=2.0,
    ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, x):
        x = self.mel_transform(x)
        x = self.amp_to_db(x)
        return x


class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        noise_dir=None,
        abnormal_rate=0.1,
        transform=None,
        snr=25,
    ):
        self.data_dir = Path(data_dir)
        self.noise_dir = noise_dir
        self.abnormal_rate = abnormal_rate
        self.transform = transform
        self.snr = snr

        self.wav2mel = Wav2Mel(
            sample_rate=48000,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        self.audio_files = list(self.data_dir.rglob("*.wav"))
        random.shuffle(self.audio_files)

        self.num_abnormal = int(len(self.audio_files) * self.abnormal_rate)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        audio_data, sample_rate = torchaudio.load(audio_file)
        label = 0

        # Add noise
        if self.noise_dir is not None and idx < self.num_abnormal:
            noise_file = random.choice(list(Path(self.noise_dir).rglob("**/*.wav")))
            audio_data, sample_rate = add_noise(audio_file, noise_file, snr=self.snr)
            label = 1

        # Convert audio to Mel spectrogram
        mel = self.wav2mel(audio_data)

        # Normalize Mel spectrogram to [0, 1]
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

        if self.transform:
            mel = self.transform(mel)

        return mel, label


if __name__ == "__main__":
    # Test the Wav2Mel class
    random_wav = torch.rand(1, int(16000 * 4.08))
    wav2mel = Wav2Mel(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=128,
        power=2.0,
    )
    mel = wav2mel(random_wav)
    print("Shape of Mel spectrogram:", mel.shape)
