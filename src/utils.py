import csv
import scienceplots
import random
import shutil
from pathlib import Path
import scienceplots

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve

plt.style.use(["science", "nature"])


def seed_everything(seed: int):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seed: {seed}")


def calc_auc(scores: np.ndarray, labels: np.ndarray, save_path=None):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    auc_score = auc(fpr, tpr)

    # save scores
    if save_path:
        scores_path = save_path.with_suffix(f".auc_{auc_score:.2f}.csv")
        with open(scores_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "score"])
            for label, score in zip(labels, scores):
                writer.writerow([label, score])

    # Save ROC curve
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % auc_score,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path.with_suffix(".pdf"))
        plt.savefig(save_path.with_suffix(".svg"))
        plt.close()
    else:
        plt.show()
    return auc_score


def calc_partial_auc(
    scores: np.ndarray, labels: np.ndarray, fpr_upper_limit: float = 0.1
):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    partial_fpr = fpr[fpr <= fpr_upper_limit]
    partial_tpr = tpr[fpr <= fpr_upper_limit]
    partial_auc_score = auc(partial_fpr, partial_tpr) / fpr_upper_limit
    return partial_auc_score


def hist_distribution(scores: np.ndarray, labels: np.ndarray, save_path=None):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    # draw histogram for label 1 and label 0
    plt.figure()
    plt.hist(scores[labels == 0], bins=100, alpha=0.5, label="Normal")
    plt.hist(scores[labels == 1], bins=100, alpha=0.5, label="Anomaly")
    plt.legend(loc="upper right")
    if save_path:
        plt.savefig(save_path.with_suffix(".pdf"))
        plt.savefig(save_path.with_suffix(".svg"))
        plt.close()
    else:
        plt.show()


def tsne_visualize(features, labels, save_path=None, title=""):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(features)
    plt.figure()
    plt.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1], label="Normal", alpha=0.5)
    plt.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1], label="Anomaly", alpha=0.5)
    plt.legend(loc="upper right")
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path.with_suffix(".pdf"))
        plt.savefig(save_path.with_suffix(".svg"))
        plt.close()
    else:
        plt.show()


def convert_to_colormap(
    images: torch.Tensor, colormap=cv2.COLORMAP_PLASMA
) -> torch.Tensor:
    """
    Convert batch of signal channel images to colormap images

    Args:
        images (torch.Tensor): (B, 1, H, W) or (1, H, W) tensor
        NOTE: images should be normalized to [0, 1]
        colormap (int): OpenCV colormap

    Returns:
        torch.Tensor: (B, 3, H, W) or (3, H, W) tensor
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images should be torch.Tensor")
    if torch.max(images) > 1 or torch.min(images) < 0:
        print("Image max: {}, min: {}".format(torch.max(images), torch.min(images)))
        raise ValueError("images should be normalized to [0, 1]")

    if images.dim() == 3:
        images = images.unsqueeze(0)

    images_np = ((images * 255).cpu().numpy()).astype(np.uint8)
    colored_images = [cv2.applyColorMap(image[0], colormap) for image in images_np]
    colored_images = np.array(colored_images).transpose(0, 3, 1, 2)
    colored_images = (
        colored_images.squeeze(0) if colored_images.shape[0] == 1 else colored_images
    )
    colored_images = colored_images.astype(np.float32) / 255
    colored_images = torch.Tensor(colored_images)
    return colored_images


def find_best_threshold(scores: np.ndarray, labels: np.ndarray):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return best_threshold


def calc_metrics(scores: np.ndarray, labels: np.ndarray, threshold=None):
    if threshold is None:
        threshold = find_best_threshold(scores, labels)
    pred_labels = (scores > threshold).astype(int)
    f1 = f1_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    return f1, precision, recall


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


if __name__ == "__main__":
    import numpy as np

    labels = np.random.randint(0, 2, 100)
    scores = np.random.rand(100)
    # evaluate(labels, scores)
    hist_distribution(scores, labels, save_path="test")
    tsne_visualize(np.random.rand(100, 10), labels, save_path="test")
