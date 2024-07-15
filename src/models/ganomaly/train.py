import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchsummary import summary
from tqdm import tqdm

from src.losses import reconstruction_loss
from src.models import BaseTrainer
from src.networks.ganomaly import Ganomaly
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
        lr=2e-4,
        num_epochs=100,
        batch_size=32,
        num_workers=4,
        recon_loss_type="l1",
        beta_rec=50,
        beta_enc=1,
        beta_adv=1,
        pretrained=None,
        device=0,
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
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.recon_loss_type = recon_loss_type
        self.beta_enc = beta_enc
        self.beta_adv = beta_adv
        self.beta_rec = beta_rec
        self.pretrained = pretrained
        self.device = torch.device(f"cuda:{device}" if device != -1 else "cpu")
        self.num_visualize = num_visualize
        self.num_row = num_row
        self.residual = residual
        self.seed = seed

        seed_everything(self.seed)

        # ========================== Print Settings ==========================
        print("Settings:")
        print(f"  Dataset: {self.dataset}")
        print(f"  Latent Dimension: {self.z_dim}")
        print(f"  Learning Rate: {self.lr}")
        print(f"  Number of Epochs: {self.num_epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Number of Workers: {self.num_workers}")
        print(f"  Reconstruction Loss Type: {self.recon_loss_type}")
        print(f"  Beta Rec: {self.beta_rec}")
        print(f"  Beta Enc: {self.beta_enc}")
        print(f"  Beta Adv: {self.beta_adv}")
        print(f"  Pretrained Model: {self.pretrained}")
        print(f"  Device: {self.device}")
        print(f"  Number of Visualization Images: {self.num_visualize}")
        print(f"  Number of Rows: {self.num_row}")
        print(f"  Residual: {self.residual}")
        print(f"  Seed: {self.seed}")

        # ========================== Build Dataset ==========================
        self.build_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_to_minus_one_one=False,
        )

        # ========================== Build Networks ==========================
        self.model = Ganomaly(
            in_channels=self.num_channels,
            image_size=self.image_size,
            latent_dim=self.z_dim,
            channels=self.channels,
            final_activation="sigmoid",
            residual=self.residual,
        ).to(self.device)

        print("Generator:")
        summary(
            self.model.generator,
            input_size=(self.num_channels, self.image_size, self.image_size),
            device=str(self.device).split(":")[0],
        )
        print("Discriminator:")
        summary(
            self.model.discriminator,
            input_size=(self.num_channels, self.image_size, self.image_size),
            device=str(self.device).split(":")[0],
        )

        # ========================== Build Optimizers ==========================
        self.optimizer_G = optim.Adam(
            self.model.generator.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.model.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

        # ========================== Load Pretrained Model ==========================
        if self.pretrained is not None:
            self.load_checkpoint(self.pretrained)

        self.loss_bce = nn.BCELoss()
        self.loss_smooth_l1 = nn.SmoothL1Loss()
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer_G": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
                "current_epoch": self.current_epoch,
                "dataset": self.dataset,
            },
            path,
        )

    def load_checkpoint(self, path):
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        self.current_epoch = checkpoint["current_epoch"]
        self.dataset = checkpoint.get("dataset", self.dataset)

    def train_one_epoch(self):
        self.model.train()

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch}/{self.num_epochs}"
        )
        for batch_idx, (real, _) in enumerate(pbar):
            real = real.to(self.device)

            # ========================== Train Generator ==========================
            self.optimizer_G.zero_grad()

            fake, latent_i, latent_o = self.model(real)
            pred_real = self.model.discriminator(real)
            pred_fake = self.model.discriminator(fake)
            enc_loss = self.loss_smooth_l1(latent_i, latent_o)
            adv_loss = self.loss_mse(pred_real, pred_fake)
            recon_loss = reconstruction_loss(
                real, fake, loss_type=self.recon_loss_type, reduction="mean"
            )

            loss_G = (
                self.beta_rec * recon_loss
                + self.beta_enc * enc_loss
                + self.beta_adv * adv_loss
            )
            loss_G.backward()
            self.optimizer_G.step()

            # ========================== Train Discriminator ==========================
            self.optimizer_D.zero_grad()

            fake, _, _ = self.model(real)
            pred_real = self.model.discriminator(real)
            pred_fake = self.model.discriminator(fake.detach())

            loss_real = self.loss_bce(
                pred_real,
                torch.ones(
                    size=(real.size(0), 1), dtype=torch.float32, device=self.device
                ),
            )
            loss_fake = self.loss_bce(
                pred_fake,
                torch.zeros(
                    size=(real.size(0), 1), dtype=torch.float32, device=self.device
                ),
            )
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            self.optimizer_D.step()

            if batch_idx % 10 == 0:
                pbar.set_postfix(
                    {
                        "Loss G": loss_G.item(),
                        "Loss D": loss_D.item(),
                        "Recon Loss": recon_loss.item(),
                        "Enc Loss": enc_loss.item(),
                        "Adv Loss": adv_loss.item(),
                    }
                )

    def train(self):
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            self.evaluate()
            self.visualize()
            self.save_checkpoint(
                Path("checkpoints")
                / "ganomaly"
                / self.dataset
                / f"epoch_{self.current_epoch}.pth"
            )

    def latents(self):
        self.model.eval()

        latents = []
        labels = []

        with torch.no_grad():
            for real_imgs, label in self.test_loader:
                real_imgs = real_imgs.to(self.device)
                _, z, _ = self.model(real_imgs)
                latents.append(z.cpu().numpy())
                labels.append(label.cpu().numpy())

        latents = np.concatenate(latents)
        labels = np.concatenate(labels)

        return latents, labels

    def evaluate(self, save_root="results"):
        self.model.eval()

        recon_losses = []
        latents = []
        anomaly_scores = []
        labels = []

        with torch.no_grad():
            for real_imgs, label in self.test_loader:
                real_imgs = real_imgs.to(self.device)
                fake_imgs, z, _ = self.model(real_imgs)
                recon_loss = (
                    reconstruction_loss(
                        fake_imgs,
                        real_imgs,
                        loss_type=self.recon_loss_type,
                        reduction="none",
                    )
                    .view(real_imgs.size(0), -1)
                    .mean(dim=1)
                )
                anomaly_score = recon_loss
                recon_losses.append(recon_loss.cpu().numpy())
                latents.append(z.cpu().numpy())
                anomaly_scores.append(anomaly_score.cpu().numpy())
                labels.append(label.cpu().numpy())

        recon_losses = np.concatenate(recon_losses)
        latents = np.concatenate(latents)
        anomaly_scores = np.concatenate(anomaly_scores)
        labels = np.concatenate(labels)

        save_root = Path(save_root) / "ganomaly" / self.dataset
        save_root.mkdir(parents=True, exist_ok=True)
        auc = calc_auc(
            anomaly_scores,
            labels,
            save_root / f"epoch_{self.current_epoch}_roc_scores.png",
        )
        print(f"AUC: {auc}")

        hist_distribution(
            anomaly_scores,
            labels,
            save_root / f"epoch_{self.current_epoch}_scores_hist.png",
        )
        hist_distribution(
            recon_losses,
            labels,
            save_root / f"epoch_{self.current_epoch}_recon_losses_hist.png",
        )
        tsne_visualize(
            latents, labels, save_root / f"epoch_{self.current_epoch}_latents.png"
        )

    def visualize(self, save_root="results"):
        self.model.eval()

        with torch.no_grad():
            normal_images = next(iter(self.normal_loader))[0][: self.num_visualize].to(
                self.device
            )
            abnormal_images = next(iter(self.abnormal_loader))[0][
                : self.num_visualize
            ].to(self.device)

            normal_rec, _, _ = self.model(normal_images)
            abnormal_rec, _, _ = self.model(abnormal_images)

            z = torch.randn(self.num_visualize, self.z_dim).to(self.device)
            sampled_images = self.model.generator.decoder(z)

        save_root = Path(save_root) / "ganomaly" / self.dataset
        save_root.mkdir(parents=True, exist_ok=True)
        vutils.save_image(
            convert_to_colormap(normal_images),
            save_root / f"epoch_{self.current_epoch}_images_normal.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(normal_rec),
            save_root / f"epoch_{self.current_epoch}_images_normal_rec.png",
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
            convert_to_colormap(abnormal_rec),
            save_root / f"epoch_{self.current_epoch}_images_abnormal_rec.png",
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
    parser = argparse.ArgumentParser(description="Train GANomaly")
    parser.add_argument("--dataset", type=str, default="ssva", help="Dataset name")
    parser.add_argument("--z_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--recon_loss_type",
        type=str,
        default="l1",
        help="Reconstruction loss type (l1, l2)",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="Pretrained model path"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument(
        "--num_visualize", type=int, default=8, help="Number of visualization images"
    )
    parser.add_argument("--num_row", type=int, default=4, help="Number of rows")
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

    trainer = Trainer(
        dataset=args.dataset,
        z_dim=args.z_dim,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        recon_loss_type=args.recon_loss_type,
        pretrained=args.pretrained,
        device=args.gpu,
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
