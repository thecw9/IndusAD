import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torchsummary import summary
from tqdm import tqdm

from src.losses import reconstruction_loss
from src.models import BaseTrainer
from src.networks.autoencoder import Autoencoder
from src.utils import (
    calc_auc,
    convert_to_colormap,
    hist_distribution,
    tsne_visualize,
    seed_everything,
)


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset,
        z_dim,
        lr=1e-3,
        batch_size=32,
        num_epochs=100,
        num_workers=4,
        recon_loss_type="l1",
        pretrained=None,
        device=torch.device("cpu"),
        num_visualize=8,
        num_row=4,
        residual=False,
        seed=-1,
    ):
        super().__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.recon_loss_type = recon_loss_type
        self.pretrained = pretrained
        self.device = device
        self.num_visualize = num_visualize
        self.num_row = num_row
        self.residual = residual
        self.seed = seed

        seed_everything(self.seed)

        # =========================== Print Settings ===========================
        print("Settings:")
        print(f"  Dataset: {dataset}")
        print(f"  Latent Dimension: {z_dim}")
        print(f"  Learning Rate: {lr}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Number of Epochs: {num_epochs}")
        print(f"  Number of Workers: {num_workers}")
        print(f"  Reconstruction Loss Type: {recon_loss_type}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Device: {device}")
        print(f"  Number of Visualize: {num_visualize}")
        print(f"  Number of Row: {num_row}")
        print(f"  Residual: {residual}")
        print(f"  Seed: {seed}")

        # =========================== Build Model ===========================
        self.build_dataloaders(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.model = Autoencoder(
            in_channels=self.num_channels,
            latent_dim=self.z_dim,
            image_size=self.image_size,
            channels=self.channels,
            residual=self.residual,
        ).to(device)
        summary(
            self.model,
            (self.num_channels, self.image_size, self.image_size),
            device=str(self.device).split(":")[0],
        )

        # =========================== Build Optimizer ===========================
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if pretrained is not None:
            self.load_checkpoint(pretrained)
            print(f"Loaded pretrained model from {pretrained}")

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            x_recon, _ = self.model(x)
            loss = reconstruction_loss(x, x_recon, loss_type=self.recon_loss_type)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                pbar.set_postfix(loss=loss.item())

    def train(self):
        for epoch in range(self.current_epoch + 1, self.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            self.evaluate()
            self.visualize()
            self.save_checkpoint(
                f"./checkpoints/autoencoder/{self.dataset}/epoch_{epoch}.pth"
            )

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "current_epoch": self.current_epoch,
                "dataset": self.dataset,
            },
            path,
        )

    def load_checkpoint(self, path):
        path = Path(path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint["current_epoch"]
        self.dataset = checkpoint.get("dataset", self.dataset)

    def latents(self):
        self.model.eval()
        latents = []
        labels = []
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                _, z = self.model(x)
                latents.append(z.cpu().numpy())
                labels.append(y)
        latents = np.concatenate(latents)
        labels = np.concatenate(labels)
        return latents, labels

    def evaluate(self, save_root="./results"):
        self.model.eval()
        recon_losses = []
        latents = []
        scores = []
        labels = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                x_recon, z = self.model(x)
                recon_loss = (
                    reconstruction_loss(
                        x, x_recon, loss_type=self.recon_loss_type, reduction="none"
                    )
                    .view(x.size(0), -1)
                    .mean(dim=1)
                )
                recon_losses.append(recon_loss.cpu().numpy())
                latents.append(z.cpu().numpy())
                scores.append(recon_loss.cpu().numpy())
                labels.append(y)
        recon_losses = np.concatenate(recon_losses)
        latents = np.concatenate(latents)
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)

        save_root = Path(save_root) / "autoencoder" / self.dataset

        auc = calc_auc(
            scores,
            labels,
            save_path=save_root / f"epoch_{self.current_epoch}_roc_scores.png",
        )
        print(f"AUC: {auc}")
        hist_distribution(
            scores,
            labels,
            save_path=save_root / f"epoch_{self.current_epoch}_dist_scores.png",
        )
        hist_distribution(
            recon_losses,
            labels,
            save_path=save_root / f"epoch_{self.current_epoch}_dist_recon_loss.png",
        )
        tsne_visualize(
            latents,
            labels,
            save_path=save_root / f"epoch_{self.current_epoch}_tsne_latents.png",
        )

    def visualize(self, save_root="./results"):
        self.model.eval()
        normal_images = next(iter(self.normal_loader))[0][: self.num_visualize].to(
            self.device
        )
        abnormal_images = next(iter(self.abnormal_loader))[0][: self.num_visualize].to(
            self.device
        )

        noise_batch = torch.randn(self.num_visualize, self.z_dim).to(self.device)
        with torch.no_grad():
            normal_recon, _ = self.model(normal_images)
            abnormal_recon, _ = self.model(abnormal_images)
            sampled_images = self.model.decode(noise_batch)

        # Save normal images
        save_root = Path(save_root) / "autoencoder" / self.dataset
        vutils.save_image(
            convert_to_colormap(normal_images),
            save_root / f"epoch_{self.current_epoch}_images_normal.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(normal_recon),
            save_root / f"epoch_{self.current_epoch}_images_normal_recon.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(abnormal_images),
            save_root / f"epoch_{self.current_epoch}_images_abnormal.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(abnormal_recon),
            save_root / f"epoch_{self.current_epoch}_images_abnormal_recon.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(sampled_images),
            save_root / f"epoch_{self.current_epoch}_images_sampled.png",
            nrow=self.num_row,
            pad_value=1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument(
        "-d", "--dataset", type=str, default="ssva", help="Dataset, available: ssva"
    )
    parser.add_argument("-z", "--z_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-w", "--num_workers", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "-r",
        "--recon_loss_type",
        type=str,
        default="mse",
        help="Reconstruction loss type",
    )
    parser.add_argument(
        "-p", "--pretrained", type=str, default=None, help="Pretrained model path"
    )
    parser.add_argument(
        "-e", "--num_epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument(
        "-v",
        "--num_visualize",
        type=int,
        default=8,
        help="Number of images to visualize",
    )
    parser.add_argument(
        "-n", "--num_row", type=int, default=4, help="Number of images in a row"
    )
    parser.add_argument(
        "-res", "--residual", action="store_true", help="Use residual connection"
    )
    parser.add_argument("-s", "--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument(
        "--eval_save",
        type=str,
        default="./experiments",
        help="Save path for evaluation",
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.gpu == -1 else torch.device(f"cuda:{args.gpu}"))
    print(f"Device: {device}")
    pretrained = args.pretrained if args.pretrained != "None" else None
    trainer = Trainer(
        dataset=args.dataset,
        z_dim=args.z_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        recon_loss_type=args.recon_loss_type,
        pretrained=pretrained,
        device=device,
        num_visualize=args.num_visualize,
        num_row=args.num_row,
        residual=args.residual,
        seed=args.seed,
    )
    if not args.eval:
        trainer.train()
    else:
        trainer.evaluate(save_root=args.eval_save)
        trainer.visualize(save_root=args.eval_save)
