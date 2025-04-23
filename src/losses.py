import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(x, x_hat, loss_type="mse", reduction="mean"):
    """
    Compute the reconstruction loss between the input and the output of the model.
    Args:
        x: input tensor
        x_hat: reconstructed tensor
        loss_type: loss function to use (mse, l1)
        reduction: reduction type (mean, sum, none)
    Returns:
        loss: reconstruction loss
    """

    x = x.view(x.size(0), -1)
    x_hat = x_hat.view(x_hat.size(0), -1)

    if loss_type == "mse":
        loss = F.mse_loss(x, x_hat, reduction="none").sum(dim=1)
    elif loss_type == "l1":
        loss = F.l1_loss(x, x_hat, reduction="none").sum(dim=1)
    else:
        raise ValueError("Invalid loss type")

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        loss = loss
    return loss


def kl_divergence(
    mu: Tensor, logvar: Tensor, mu_prior=0.0, logvar_prior=0.0, reduction="sum"
) -> Tensor:
    if not isinstance(mu_prior, Tensor):
        mu_prior = torch.tensor(mu_prior, device=mu.device)
    if not isinstance(logvar_prior, Tensor):
        logvar_prior = torch.tensor(logvar_prior, device=mu.device)
    kl = -0.5 * (
        1
        + logvar
        - logvar_prior
        - logvar.exp() / torch.exp(logvar_prior)
        - (mu - mu_prior).pow(2) / torch.exp(logvar_prior)
    ).sum(1)

    if reduction == "mean":
        kl = torch.mean(kl)
    elif reduction == "sum":
        kl = torch.sum(kl)
    elif reduction == "none":
        kl = kl
    return kl


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    x_hat = torch.randn(2, 3, 32, 32)
    loss = reconstruction_loss(x, x_hat, loss_type="mse", reduction="none")
    print(loss.shape)
