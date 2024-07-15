import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torchsummary import summary
from tqdm import tqdm

from src.losses import kl_divergence, reconstruction_loss
from src.models import BaseTrainer
from src.networks.vae import VAE
from src.utils import (
    calc_auc,
    convert_to_colormap,
    hist_distribution,
    seed_everything,
    tsne_visualize,
)


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset="ssva",
        z_dim=32,
        lr=1e-3,
        beta_kl=1.0,
        num_epochs=100,
        batch_size=32,
        num_workers=4,
        recon_loss_type="l1",
        pretrained=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_visualize=8,
        num_row=4,
        residual=False,
        seed=42,
    ):
        super(Trainer, self).__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        self.lr = lr
        self.beta_kl = beta_kl
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
        print("Settings: ")
        print(f"  Dataset: {dataset}")
        print(f"  Latent Dimension: {z_dim}")
        print(f"  Learning Rate: {lr}")
        print(f"  Beta KL: {beta_kl}")
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

        # =========================== Build DataLoader ===========================
        self.build_dataloaders(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

        # =========================== Build Model ===========================
        self.model = VAE(
            in_channels=self.num_channels,
            latent_dim=self.z_dim,
            image_size=self.image_size,
            channels=self.channels,
            final_activation="sigmoid",
            residual=self.residual,
        ).to(device)
        summary(
            self.model,
            (self.num_channels, self.image_size, self.image_size),
            device=str(self.device).split(":")[0],
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if pretrained is not None:
            self.load_checkpoint(pretrained)
            print(f"Pretrained model loaded from {pretrained}")

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "dataset": self.dataset,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        path = Path(path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint["current_epoch"]
        self.dataset = checkpoint.get("dataset", self.dataset)

    def train_one_epoch(self, train_loader):
        self.model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, _, mu, logvar = self.model(data)
            recon_loss = reconstruction_loss(
                data, recon_batch, loss_type=self.recon_loss_type, reduction="mean"
            )
            kl_loss = kl_divergence(mu, logvar, reduction="mean")
            loss = recon_loss + self.beta_kl * kl_loss
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                pbar.set_postfix(
                    {
                        "Recon Loss": recon_loss.item(),
                        "KL Loss": kl_loss.item(),
                        "Total Loss": loss.item(),
                    }
                )

    def train(self):
        for epoch in range(self.current_epoch + 1, self.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch(self.train_loader)

            self.evaluate()
            self.visualize()
            self.save_checkpoint(f"./checkpoints/vae/{self.dataset}/epoch_{epoch}.pth")

    def latents(self):
        self.model.eval()
        latents = []
        labels = []
        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.to(self.device)
                _, z, _, _ = self.model(data)
                latents.append(z.cpu().numpy())
                labels.append(label.cpu().numpy())
        latents = np.concatenate(latents, axis=0)
        labels = np.concatenate(labels, axis=0)

        return latents, labels

    def evaluate(self, save_root="./results"):
        self.model.eval()
        recon_losses = []
        latents = []
        kl_divergences = []
        anomaly_scores = []
        labels = []
        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.to(self.device)
                recon_batch, z, mu, logvar = self.model(data)
                recon_loss = (
                    reconstruction_loss(
                        data,
                        recon_batch,
                        loss_type=self.recon_loss_type,
                        reduction="none",
                    )
                    .view(data.size(0), -1)
                    .mean(1)
                )
                kl_loss = (
                    kl_divergence(mu, logvar, reduction="none")
                    .view(data.size(0), -1)
                    .mean(1)
                )
                anomay_score = recon_loss
                recon_losses.append(recon_loss.cpu().numpy())
                latents.append(z.cpu().numpy())
                kl_divergences.append(kl_loss.cpu().numpy())
                anomaly_scores.append(anomay_score.cpu().numpy())
                labels.append(label.cpu().numpy())
        recon_losses = np.concatenate(recon_losses, axis=0)
        latents = np.concatenate(latents, axis=0)
        kl_divergences = np.concatenate(kl_divergences, axis=0)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        root_dir = Path(save_root) / "vae" / self.dataset
        root_dir.mkdir(parents=True, exist_ok=True)
        auc = calc_auc(
            anomaly_scores,
            labels,
            save_path=root_dir / f"epoch_{self.current_epoch}_roc_scores.png",
        )
        print(f"AUC: {auc}")
        hist_distribution(
            anomaly_scores,
            labels,
            save_path=root_dir / f"epoch_{self.current_epoch}_dist_scores.png",
        )
        hist_distribution(
            kl_divergences,
            labels,
            save_path=root_dir / f"epoch_{self.current_epoch}_dist_kl_divergences.png",
        )
        hist_distribution(
            recon_losses,
            labels,
            save_path=root_dir / f"epoch_{self.current_epoch}_dist_recon_losses.png",
        )
        tsne_visualize(
            latents,
            labels,
            save_path=root_dir / f"epoch_{self.current_epoch}_tsne_latent.png",
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
            normal_recon, *_ = self.model(normal_images)
            abnormal_recon, *_ = self.model(abnormal_images)
            sampled_images = self.model.decode(noise_batch)

        # Save normal images
        root_dir = Path(save_root) / "vae" / self.dataset
        root_dir.mkdir(parents=True, exist_ok=True)
        vutils.save_image(
            convert_to_colormap(normal_images),
            root_dir / f"epoch_{self.current_epoch}_images_normal.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(normal_recon),
            root_dir / f"epoch_{self.current_epoch}_images_normal_recon.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(abnormal_images),
            root_dir / f"epoch_{self.current_epoch}_images_abnormal.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(abnormal_recon),
            root_dir / f"epoch_{self.current_epoch}_images_abnormal_recon.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(sampled_images),
            root_dir / f"epoch_{self.current_epoch}_images_sampled.png",
            nrow=self.num_row,
            pad_value=1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument(
        "-d", "--dataset", type=str, default="ssva", help="Dataset, available: ssva"
    )
    parser.add_argument("-z", "--z_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta_kl", type=float, default=1.0, help="Beta KL")
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
        "--residual", action="store_true", help="Use residual connection"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument(
        "--eval_save",
        type=str,
        default="./experiments",
        help="Save path for evaluation",
    )

    args = parser.parse_args()

    device = torch.device("cpu" if args.gpu == -1 else torch.device(f"cuda:{args.gpu}"))
    pretrained = args.pretrained if args.pretrained != "None" else None

    trainer = Trainer(
        dataset=args.dataset,
        z_dim=args.z_dim,
        lr=args.lr,
        beta_kl=args.beta_kl,
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
