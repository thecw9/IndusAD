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
from src.networks.wgan_gp import Discriminator, Encoder, Generator
from src.utils import (
    calc_auc,
    convert_to_colormap,
    hist_distribution,
    seed_everything,
    tsne_visualize,
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
        z_dim=32,
        lr=1e-3,
        num_epochs=100,
        num_encoder_train=1,
        train_encoder_interval=1,
        batch_size=32,
        num_workers=4,
        recon_loss_type="mse",
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
        self.num_workers = num_workers
        self.num_encoder_train = num_encoder_train
        self.train_encoder_interval = train_encoder_interval
        self.num_epochs = num_epochs
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
        print(f"  Number of Workers: {num_workers}")
        print(f"  Number of Encoder Train: {num_encoder_train}")
        print(f"  Number of Epochs: {num_epochs}")
        print(f"  Reconstruction Loss Type: {recon_loss_type}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Device: {device}")
        print(f"  Number of Visualize: {num_visualize}")
        print(f"  Number of Row: {num_row}")
        print(f"  Residual: {residual}")

        # ========================== Build Dataset ==========================
        self.build_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_to_minus_one_one=True,
        )

        # ========================== Build Model ==========================
        self.netD = Discriminator(
            in_channels=self.num_channels,
            image_size=self.image_size,
            channels=self.channels,
        ).to(device)
        self.netG = Generator(
            out_channels=self.num_channels,
            image_size=self.image_size,
            latent_dim=z_dim,
            channels=self.channels[::-1],
            conv_input_size=self.netD.conv_output_size,
            final_activation="tanh",
            residual=self.residual,
        ).to(device)
        self.encoder = Encoder(
            in_channels=self.num_channels,
            image_size=self.image_size,
            channels=self.channels,
            latent_dim=z_dim,
        ).to(device)
        summary(
            self.netG,
            (z_dim,),
            device=str(self.device).split(":")[0],
        )
        summary(
            self.netD,
            (self.num_channels, self.image_size, self.image_size),
            device=str(self.device).split(":")[0],
        )

        # weights initialization
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)

        # ========================== Build Optimizer ==========================
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerE = optim.Adam(
            self.encoder.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        # ========================== Load Pretrained Model ==========================
        if pretrained is not None:
            self.load_checkpoint(pretrained)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "netD": self.netD.state_dict(),
                "netG": self.netG.state_dict(),
                "encoder": self.encoder.state_dict(),
                "optimizerD": self.optimizerD.state_dict(),
                "optimizerG": self.optimizerG.state_dict(),
                "optimizerE": self.optimizerE.state_dict(),
                "current_epoch": self.current_epoch,
                "dataset": self.dataset,
            },
            path,
        )

    def load_checkpoint(self, path):
        path = Path(path)
        checkpoint = torch.load(path)
        self.netD.load_state_dict(checkpoint["netD"])
        self.netG.load_state_dict(checkpoint["netG"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.optimizerD.load_state_dict(checkpoint["optimizerD"])
        self.optimizerG.load_state_dict(checkpoint["optimizerG"])
        self.optimizerE.load_state_dict(checkpoint["optimizerE"])
        self.current_epoch = checkpoint["current_epoch"]
        self.dataset = checkpoint.get("dataset", self.dataset)

    def gp(self, real_imgs, fake_imgs):
        """
        Gradient penalty.
        """
        batch_size = real_imgs.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1).to(self.device)
        fake_imgs = fake_imgs.detach()
        x_hat = epsilon * real_imgs + (1 - epsilon) * fake_imgs
        x_hat.requires_grad_(True)
        hat_logits = self.netD(x_hat)
        grad = torch.autograd.grad(
            outputs=hat_logits,
            inputs=x_hat,
            grad_outputs=torch.ones_like(hat_logits),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad = grad.view(grad.size(0), -1)
        grad_norm = grad.norm(2, dim=1)
        gp = torch.mean((grad_norm - 1) ** 2)
        return gp

    def train_one_epoch(self):
        self.netD.train()
        self.netG.train()
        self.encoder.eval()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(self.device)
            batch_size = real_imgs.size(0)

            # ========================== Train Discriminator ==========================
            self.optimizerD.zero_grad()
            z = torch.randn(batch_size, self.z_dim).to(self.device)
            fake_imgs = self.netG(z)
            real_logits = self.netD(real_imgs)
            fake_logits = self.netD(fake_imgs.detach())
            d_loss = -torch.mean(real_logits) + torch.mean(fake_logits)
            gp = self.gp(real_imgs, fake_imgs)
            d_loss += 10 * gp
            d_loss.backward()
            self.optimizerD.step()

            # ========================== Train Generator ==========================
            self.optimizerG.zero_grad()
            fake_logits = self.netD(fake_imgs)
            g_loss = -torch.mean(fake_logits)
            g_loss.backward()
            self.optimizerG.step()

            # ========================== Tqdm ==========================
            if batch_idx % 10 == 0:
                pbar.set_postfix(
                    {
                        "D Loss": d_loss.item(),
                        "G Loss": g_loss.item(),
                        "GP": gp.item(),
                    }
                )

    def train_encoder(self):
        self.encoder.train()
        self.netD.eval()
        self.netG.eval()

        pbar = tqdm(range(self.num_encoder_train), desc="Train Encoder")
        for _ in pbar:
            for batch_idx, (real_imgs, _) in enumerate(self.train_loader):
                real_imgs = real_imgs.to(self.device)

                self.optimizerE.zero_grad()
                z = self.encoder(real_imgs)
                fake_imgs = self.netG(z)
                loss = reconstruction_loss(
                    fake_imgs, real_imgs, loss_type=self.recon_loss_type
                )
                loss.backward()
                self.optimizerE.step()

                if batch_idx % 10 == 0:
                    pbar.set_postfix({"Recon Loss": loss.item()})

    def train(self):
        for _ in range(self.current_epoch, self.num_epochs):
            self.train_one_epoch()
            if self.current_epoch % self.train_encoder_interval == 0:
                self.train_encoder()
            self.current_epoch += 1

            self.evaluate()
            self.visualize()
            self.save_checkpoint(
                f"checkpoints/fanogan/{self.dataset}/epoch_{self.current_epoch}.pth"
            )

    def latents(self):
        self.netD.eval()
        self.netG.eval()
        self.encoder.eval()

        latents = []
        labels = []

        with torch.no_grad():
            for real_imgs, label in self.train_loader:
                real_imgs = real_imgs.to(self.device)
                z = self.encoder(real_imgs)
                latents.append(z.cpu().numpy())
                labels.append(label.cpu().numpy())

        latents = np.concatenate(latents)
        labels = np.concatenate(labels)
        return latents, labels

    def evaluate(self, save_root="results"):
        self.netD.eval()
        self.netG.eval()
        self.encoder.eval()

        recon_losses = []
        latents = []
        anomaly_scores = []
        labels = []

        with torch.no_grad():
            for real_imgs, label in self.test_loader:
                real_imgs = real_imgs.to(self.device)
                z = self.encoder(real_imgs)
                fake_imgs = self.netG(z)
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

        save_root = Path(save_root) / "fanogan" / self.dataset
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
            save_root / f"epoch_{self.current_epoch}_dist_scores.png",
        )
        hist_distribution(
            recon_losses,
            labels,
            save_root / f"epoch_{self.current_epoch}_dist_recon_losses.png",
        )
        tsne_visualize(
            latents, labels, save_root / f"epoch_{self.current_epoch}_tsne_latents.png"
        )

    def visualize(self, save_root="results"):
        self.netD.eval()
        self.netG.eval()
        self.encoder.eval()

        with torch.no_grad():
            normal_imgs = next(iter(self.normal_loader))[0][: self.num_visualize].to(
                self.device
            )
            abnormal_imgs = next(iter(self.abnormal_loader))[0][
                : self.num_visualize
            ].to(self.device)

            normal_recon = self.netG(self.encoder(normal_imgs))
            abnormal_recon = self.netG(self.encoder(abnormal_imgs))

            z = torch.randn(self.num_visualize, self.z_dim).to(self.device)
            sampled_imgs = self.netG(z)

        # Save images
        save_root = Path(save_root) / "fanogan" / self.dataset
        save_root.mkdir(parents=True, exist_ok=True)
        vutils.save_image(
            convert_to_colormap((normal_imgs + 1) / 2),
            save_root / f"epoch_{self.current_epoch}_images_normal.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap((abnormal_imgs + 1) / 2),
            save_root / f"epoch_{self.current_epoch}_images_abnormal.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap((normal_recon + 1) / 2),
            save_root / f"epoch_{self.current_epoch}_images_normal_recon.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap((abnormal_recon + 1) / 2),
            save_root / f"epoch_{self.current_epoch}_images_abnormal_recon.png",
            nrow=self.num_row,
            pad_value=1,
        )
        vutils.save_image(
            convert_to_colormap((sampled_imgs + 1) / 2),
            save_root / f"epoch_{self.current_epoch}_images_sampled.png",
            nrow=self.num_row,
            pad_value=1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ssva")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_encoder_train", type=int, default=1)
    parser.add_argument("--train_encoder_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--recon_loss_type", type=str, default="mse")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument("--num_visualize", type=int, default=8)
    parser.add_argument("--num_row", type=int, default=4)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument(
        "--eval_save",
        type=str,
        default="./experiments",
        help="Save path for evaluation",
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.gpu == -1 else torch.device(f"cuda:{args.gpu}"))

    trainer = Trainer(
        dataset=args.dataset,
        z_dim=args.z_dim,
        lr=args.lr,
        num_epochs=args.num_epochs,
        num_encoder_train=args.num_encoder_train,
        train_encoder_interval=args.train_encoder_interval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        recon_loss_type=args.recon_loss_type,
        pretrained=args.pretrained,
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
