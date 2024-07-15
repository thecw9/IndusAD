# standard

import argparse
import os
from pathlib import Path

import numpy as np

# imports
# torch and friends
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import torchvision.utils as vutils
import torch.nn as nn
from tqdm import tqdm

from src.models import BaseTrainer
from src.networks.lraad import LRAAD
from src.losses import kl_divergence, reconstruction_loss
from src.utils import (
    calc_auc,
    hist_distribution,
    seed_everything,
    tsne_visualize,
    convert_to_colormap,
    reparameterize,
)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Trainer(BaseTrainer):
    def __init__(
        self,
        dataset="ssva",
        z_dim=128,
        lr_e=2e-4,
        lr_d=2e-4,
        batch_size=128,
        num_workers=4,
        num_epochs=250,
        num_vae_iter=0,
        recon_loss_type="mse",
        beta_kl=1.0,
        beta_rec=1.0,
        beta_adv=256,
        seed=-1,
        pretrained=None,
        device=torch.device("cpu"),
        num_row=4,
        gamma_r=1e-8,
        final_activation="none",
        threshold_low=10,
        threshold_high=100,
        decay=0.01,
    ):
        super(Trainer, self).__init__()

        self.dataset = dataset
        self.z_dim = z_dim
        self.lr_e = lr_e
        self.lr_d = lr_d
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.num_vae_iter = num_vae_iter
        self.recon_loss_type = recon_loss_type
        self.beta_kl = beta_kl
        self.beta_rec = beta_rec
        self.beta_adv = beta_adv
        self.seed = seed
        self.pretrained = pretrained
        self.device = device
        self.num_row = num_row
        self.gamma_r = gamma_r
        self.final_activation = final_activation
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.decay = decay

        self.curent_iter_num = 0

        seed_everything(self.seed)

        # ======================= Print Settings =======================
        print("Settings:")
        print(f"  Dataset: {self.dataset}")
        print(f"  Latent Dimension: {self.z_dim}")
        print(f"  Learning Rate (Encoder): {self.lr_e}")
        print(f"  Learning Rate (Decoder): {self.lr_d}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Number of Workers: {self.num_workers}")
        print(f"  Number of Epochs: {self.num_epochs}")
        print(f"  Number of VAE Iterations: {self.num_vae_iter}")
        print(f"  Reconstruction Loss Type: {self.recon_loss_type}")
        print(f"  KL Beta: {self.beta_kl}")
        print(f"  Reconstruction Beta: {self.beta_rec}")
        print(f"  Adversarial Beta: {self.beta_adv}")
        print(f"  Seed: {self.seed}")
        print(f"  Pretrained Model: {self.pretrained}")
        print(f"  Device: {self.device}")
        print(f"  Number of Rows: {self.num_row}")
        print(f"  Gamma R: {self.gamma_r}")
        print(f"  Final Activation: {self.final_activation}")
        print(f"  Threshold Low: {self.threshold_low}")
        print(f"  Threshold High: {self.threshold_high}")
        print(f"  Decay: {self.decay}")

        # ======================= Build Model =======================
        self.build_dataloaders(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.model = LRAAD(
            in_channels=self.num_channels,
            latent_dim=self.z_dim,
            channels=self.channels,
            image_size=self.image_size,
            final_activation=self.final_activation,
        ).to(device)

        # ======================= Optimizer =======================
        self.optimizer_e = optim.Adam(self.model.encoder.parameters(), lr=lr_e)
        self.optimizer_g = optim.Adam(self.model.decoder.parameters(), lr=lr_d)

        self.e_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer_e, milestones=(350,), gamma=0.1
        )
        self.d_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer_g, milestones=(350,), gamma=0.1
        )

        if pretrained is not None:
            self.load_checkpoint(pretrained)
            print(f"Pretrained model loaded from {pretrained}")
        summary(
            self.model,
            (self.num_channels, self.image_size, self.image_size),
            device=str(self.device).split(":")[0],
        )

        self.scale = 1.0 / (1 * 128 * 128)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer_e": self.optimizer_e.state_dict(),
                "optimizer_d": self.optimizer_g.state_dict(),
                "e_scheduler": self.e_scheduler.state_dict(),
                "d_scheduler": self.d_scheduler.state_dict(),
                "current_epoch": self.current_epoch,
                "dataset": self.dataset,
            },
            path,
        )

    def load_checkpoint(self, path):
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer_e.load_state_dict(checkpoint["optimizer_e"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_d"])
        self.e_scheduler.load_state_dict(checkpoint["e_scheduler"])
        self.d_scheduler.load_state_dict(checkpoint["d_scheduler"])
        self.current_epoch = checkpoint["current_epoch"]
        self.dataset = checkpoint.get("dataset", self.dataset)

    def kl_threshold(
        self, current_iter, threshold_low=10, threshold_high=100, decay=0.01
    ):
        threshold = (threshold_high - threshold_low) * torch.exp(
            -torch.tensor(current_iter * decay)
        ) + threshold_low
        return threshold

    def train_one_epoch(self):
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for _, (real, _) in enumerate(pbar):
            self.curent_iter_num += 1
            # --------------train------------
            if self.curent_iter_num < self.num_vae_iter:
                real = real.to(self.device)

                # =========== Update E, D ================

                real_mu, real_logvar, z, rec = self.model(real)

                rec_loss = reconstruction_loss(
                    real, rec, loss_type=self.recon_loss_type, reduction="mean"
                )
                kl = kl_divergence(real_mu, real_logvar, reduction="mean")

                loss = self.beta_rec * rec_loss + self.beta_kl * kl

                self.optimizer_g.zero_grad()
                self.optimizer_e.zero_grad()
                loss.backward()
                self.optimizer_e.step()
                self.optimizer_g.step()

                pbar.set_postfix(
                    rec_loss=rec_loss.data.cpu().item(), kl=kl.data.cpu().item()
                )
            else:
                noise_batch = torch.randn(size=(real.size(0), self.z_dim)).to(
                    self.device
                )

                real = real.to(self.device)

                # =========== Update E ================
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                for param in self.model.decoder.parameters():
                    param.requires_grad = False

                fake = self.model.sample(noise_batch)

                real_mu, real_logvar, z, rec = self.model(real)
                rec_mu, rec_logvar, z_rec, rec_rec = self.model(rec.detach())
                fake_mu, fake_logvar, z_fake, rec_fake = self.model(fake.detach())

                lossE_kl_real = kl_divergence(real_mu, real_logvar, reduction="mean")
                lossE_kl_rec = kl_divergence(rec_mu, rec_logvar, reduction="mean")
                lossE_kl_fake = kl_divergence(fake_mu, fake_logvar, reduction="mean")

                lossE_rec_real = reconstruction_loss(
                    real, rec, loss_type=self.recon_loss_type, reduction="mean"
                )

                kl_threshold = self.kl_threshold(
                    self.curent_iter_num,
                    threshold_low=self.threshold_low,
                    threshold_high=self.threshold_high,
                    decay=self.decay,
                )

                lossE = (
                    lossE_rec_real * self.beta_rec
                    + lossE_kl_real * self.beta_kl
                    + (
                        F.relu(kl_threshold - lossE_kl_rec)
                        + F.relu(kl_threshold - lossE_kl_fake)
                    )
                    * 0.5
                    * self.beta_kl
                )

                self.optimizer_e.zero_grad()
                lossE.backward()
                self.optimizer_e.step()

                # ========= Update D ==================
                for _ in range(5):
                    for param in self.model.encoder.parameters():
                        param.requires_grad = False
                    for param in self.model.decoder.parameters():
                        param.requires_grad = True

                    rec = self.model.decoder(z.detach())
                    fake = self.model.sample(noise_batch)

                    rec_mu, rec_logvar = self.model.encode(rec)
                    fake_mu, fake_logvar = self.model.encode(fake)

                    z_rec = reparameterize(rec_mu, rec_logvar)
                    z_fake = reparameterize(fake_mu, fake_logvar)

                    rec_rec = self.model.decode(z_rec.detach())
                    rec_fake = self.model.decode(z_fake.detach())

                    lossG_rec_real = reconstruction_loss(
                        real,
                        rec,
                        loss_type=self.recon_loss_type,
                        reduction="mean",
                    )

                    lossG_kl_rec = kl_divergence(rec_mu, rec_logvar, reduction="mean")
                    lossG_kl_fake = kl_divergence(
                        fake_mu, fake_logvar, reduction="mean"
                    )

                    lossG = (
                        self.beta_rec * lossG_rec_real
                        + self.beta_adv * (lossG_kl_rec + lossG_kl_fake) * 0.5
                    )

                    self.optimizer_g.zero_grad()
                    lossG.backward()
                    self.optimizer_g.step()
                    if torch.isnan(lossG) or torch.isnan(lossE):
                        raise SystemError

                pbar.set_postfix(
                    recon_loss=lossE_rec_real.data.cpu().item(),
                    kl_real=lossE_kl_real.data.cpu().item(),
                    kl_rec=lossE_kl_rec.mean().data.cpu().item(),
                )
        self.e_scheduler.step()
        self.d_scheduler.step()
        pbar.close()

    def latents(self):
        # self.model.eval()
        latents = []
        labels = []
        for images, label in self.train_loader:
            with torch.no_grad():
                images = images.to(device)
                mu, logvar, z, rec = self.model(images)
                latents.append(z.cpu().numpy())
                labels.append(label)
        latents = np.concatenate(latents, axis=0)
        labels = np.concatenate(labels, axis=0)
        return latents, labels

    def evaluate(self, save_root="results"):
        # self.model.eval()
        test_loader = self.test_loader
        latents = []
        kl_divergences = []
        rec_losses = []
        scores = []
        labels = []
        for images, label in test_loader:
            with torch.no_grad():
                images = images.to(device)
                mu, logvar, z, rec = self.model(images)
                # compute anomaly score
                loss_rec = reconstruction_loss(
                    images, rec, loss_type=self.recon_loss_type, reduction="none"
                )
                kl = kl_divergence(mu, logvar, reduction="none")
                loss = self.beta_rec * loss_rec + self.beta_kl * kl

                latents.append(z.cpu().numpy())
                kl_divergences.append(kl.cpu().numpy())
                rec_losses.append(loss_rec.cpu().numpy())
                scores.append(loss.cpu().numpy())
                labels.append(label)
        latents = np.concatenate(latents, axis=0)
        kl_divergences = np.concatenate(kl_divergences, axis=0)
        rec_losses = np.concatenate(rec_losses, axis=0)
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        save_root = Path(save_root) / "lraad" / self.dataset
        os.makedirs(save_root, exist_ok=True)

        recon_auc = calc_auc(
            rec_losses,
            labels,
            save_path=Path(save_root)
            / f"epoch_{self.current_epoch}_roc_recon_losses.jpg",
        )
        print(f"Epoch {self.current_epoch} Recon AUC: {recon_auc}")

        kl_auc = calc_auc(
            kl_divergences,
            labels,
            save_path=Path(save_root)
            / f"epoch_{self.current_epoch}_roc_kl_divergences.jpg",
        )
        print(f"Epoch {self.current_epoch} KL AUC: {kl_auc}")

        auc = calc_auc(
            scores,
            labels,
            save_path=Path(save_root) / f"epoch_{self.current_epoch}_roc_scores.jpg",
        )
        print(f"Epoch {self.current_epoch} AUC: {auc}")

        hist_distribution(
            kl_divergences,
            labels,
            save_path=Path(save_root)
            / f"epoch_{self.current_epoch}_dist_kl_divergences.jpg",
        )
        hist_distribution(
            rec_losses,
            labels,
            save_path=Path(save_root)
            / f"epoch_{self.current_epoch}_dist_rec_losses.jpg",
        )
        hist_distribution(
            scores,
            labels,
            save_path=Path(save_root) / f"epoch_{self.current_epoch}_dist_score.jpg",
        )
        tsne_visualize(
            latents,
            labels,
            save_path=Path(save_root) / f"epoch_{self.current_epoch}_tsne_latent.jpg",
        )

    def visualize(self, save_root="results"):
        # self.model.eval()
        num_visualize = 8
        normal_images = next(iter(self.normal_loader))[0][:num_visualize].to(device)
        abnormal_images = next(iter(self.abnormal_loader))[0][:num_visualize].to(device)
        noise_batch = torch.randn(size=(num_visualize, self.z_dim)).to(device)

        with torch.no_grad():
            _, _, _, normal_rec = self.model(normal_images)
            _, _, _, abnormal_rec = self.model(abnormal_images)
            sampled = self.model.sample(noise_batch)

        # save images
        save_root = Path(save_root) / "lraad" / self.dataset
        vutils.save_image(
            convert_to_colormap(normal_images),
            Path(save_root) / f"epoch_{self.current_epoch}_images_normal.jpg",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(abnormal_images),
            Path(save_root) / f"epoch_{self.current_epoch}_images_abnormal.jpg",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(
                (normal_rec - torch.min(normal_rec))
                / (torch.max(normal_rec) - torch.min(normal_rec))
            ),
            Path(save_root) / f"epoch_{self.current_epoch}_images_normal_recon.jpg",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(
                (abnormal_rec - torch.min(abnormal_rec))
                / (torch.max(abnormal_rec) - torch.min(abnormal_rec))
            ),
            Path(save_root) / f"epoch_{self.current_epoch}_images_abnormal_recon.jpg",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap(
                (sampled - torch.min(sampled))
                / (torch.max(sampled) - torch.min(sampled))
            ),
            Path(save_root) / f"epoch_{self.current_epoch}_images_sampled.jpg",
            nrow=self.num_row,
            pad_value=1,
        )

    def train(self):
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            kl_threshold = self.kl_threshold(
                self.curent_iter_num,
                threshold_low=self.threshold_low,
                threshold_high=self.threshold_high,
                decay=self.decay,
            )
            print(f"KL Threshold: {kl_threshold}")

            self.save_checkpoint(
                f"checkpoints/lraad/{self.dataset}/epoch_{self.current_epoch}.pth"
            )
            self.evaluate()
            self.visualize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ssva")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_vae_iter", type=int, default=0)
    parser.add_argument("--lr_e", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--beta_kl", type=float, default=1.0)
    parser.add_argument("--beta_adv", type=float, default=256)
    parser.add_argument("--beta_rec", type=float, default=1.0)
    parser.add_argument("--num_row", type=int, default=4)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--gamma_r", type=float, default=1e-8)
    parser.add_argument("--final_activation", type=str, default="none")
    parser.add_argument("--threshold_low", type=float, default=10)
    parser.add_argument("--threshold_high", type=float, default=100)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument(
        "--eval_save",
        type=str,
        default="./experiments",
        help="Save path for evaluation",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        dataset=args.dataset,
        z_dim=args.z_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        num_vae_iter=args.num_vae_iter,
        beta_kl=args.beta_kl,
        beta_rec=args.beta_rec,
        beta_adv=args.beta_adv,
        device=device,
        lr_e=args.lr_e,
        lr_d=args.lr_d,
        pretrained=args.pretrained,
        seed=args.seed,
        num_row=args.num_row,
        gamma_r=args.gamma_r,
        final_activation=args.final_activation,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        decay=args.decay,
    )
    if not args.eval:
        trainer.train()
    else:
        trainer.evaluate(save_root=args.eval_save)
        trainer.visualize(save_root=args.eval_save)
