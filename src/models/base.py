from typing import Tuple
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import AudioDataset
from torchvision.datasets import MNIST


class BaseTrainer:
    def __init__(self):
        self.current_epoch = 0

    def train(self):
        raise NotImplementedError

    def load_checkpoint(self, path):
        raise NotImplementedError

    def save_checkpoint(self, path):
        raise NotImplementedError

    def latents(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def evaluate(self, save_root):
        raise NotImplementedError

    def visualize(self, save_root):
        raise NotImplementedError

    def build_dataloaders(
        self,
        dataset="ssva",
        batch_size=32,
        num_workers=4,
        transform_to_minus_one_one=False,
    ):
        if dataset == "ssva":
            self.build_ssva_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                snr=25,
                trasnform_to_minus_one_one=transform_to_minus_one_one,
            )
        elif dataset == "ssva1":
            self.build_ssva_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                snr=25,
                trasnform_to_minus_one_one=transform_to_minus_one_one,
            )
        elif dataset == "ssva2":
            self.build_ssva_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                snr=24,
                trasnform_to_minus_one_one=transform_to_minus_one_one,
            )
        elif dataset == "ssva3":
            self.build_ssva_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                snr=23,
                trasnform_to_minus_one_one=transform_to_minus_one_one,
            )
        elif dataset == "ssva4":
            self.build_ssva_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                snr=22,
                trasnform_to_minus_one_one=transform_to_minus_one_one,
            )
        elif dataset == "ssva5":
            self.build_ssva_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                snr=21,
                trasnform_to_minus_one_one=transform_to_minus_one_one,
            )
        elif dataset == "ssva6":
            self.build_ssva_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                snr=31,
                trasnform_to_minus_one_one=transform_to_minus_one_one,
            )
        elif dataset == "mnist":
            self.build_mnist_dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                transform_to_minus_one_one=transform_to_minus_one_one,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def build_ssva_dataloaders(
        self,
        batch_size=32,
        num_workers=4,
        snr=25,
        trasnform_to_minus_one_one=False,
        train_dir="./data/ssva/train",
        noise_dir="./data/ssva/noise",
        test_dir="./data/ssva/test",
    ):
        self.num_channels = 1
        self.channels = [64, 128, 256, 512, 512]
        self.image_size = 128
        if trasnform_to_minus_one_one:
            transform = transforms.Compose(
                [
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            transform = None

        train_dataset = AudioDataset(
            data_dir=train_dir,
            noise_dir=None,
            abnormal_rate=0,
            transform=transform,
            snr=snr,
        )
        test_dataset = AudioDataset(
            data_dir=test_dir,
            noise_dir=noise_dir,
            abnormal_rate=0.4,
            transform=transform,
            snr=snr,
        )
        normal_dataset = AudioDataset(
            data_dir=test_dir,
            noise_dir=None,
            abnormal_rate=0,
            transform=transform,
            snr=snr,
        )
        abnormal_dataset = AudioDataset(
            data_dir=test_dir,
            noise_dir=noise_dir,
            abnormal_rate=1,
            transform=transform,
            snr=snr,
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.normal_loader = DataLoader(
            normal_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.abnormal_loader = DataLoader(
            abnormal_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def build_mnist_dataloaders(
        self, batch_size=32, num_workers=4, transform_to_minus_one_one=False
    ):
        self.num_channels = 1
        self.channels = [32, 64, 128, 256, 512]
        self.image_size = 128
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = MNIST(
            root="./data/mnist", train=True, transform=transform, download=True
        )
        test_dataset = MNIST(
            root="./data/mnist", train=False, transform=transform, download=True
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.abnormal_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.normal_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )


if __name__ == "__main__":
    trainer = BaseTrainer()
    trainer.build_dataloaders(
        dataset="mnist", batch_size=32, num_workers=4, transform_to_minus_one_one=False
    )
    image = trainer.train_loader.dataset[0]
    print(max(image[0].flatten()))
    print(min(image[0].flatten()))
