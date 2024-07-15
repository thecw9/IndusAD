from pathlib import Path
import subprocess
import os
import sys
import json
import numpy as np
from typing import Tuple
from sklearn.metrics import roc_curve, auc
import scienceplots
import pandas as pd
import matplotlib.pyplot as plt

os.environ["PYTHONPATH"] = os.getcwd()

plt.style.use(["science", "nature"])
# plt.rcParams.update({"font.size": 20})
# plt.rcParams["lines.linewidth"] = 1.5

train_args = {
    "autoencoder": ["--num_epochs 50", "--recon_loss_type mse"],
    "vae": ["--num_epochs 50", "--beta_kl 1", "--z_dim 32"],
    "fanogan": ["--num_epochs 100"],
    "ganomaly": ["--num_epochs 100"],
    "lraad": [
        "--num_epochs 100",
        "--beta_kl 1",
        "--num_vae_iter 10",
        "--final_activation none",
        "--z_dim 32",
        "--threshold_low 100",
        "--threshold_high 100",
        "--decay 0.0001",
        "--beta_adv 1",
    ],
    "improved_lraad": ["--num_epochs 100", "--beta_kl 1"],
}

eval_config = {
    "ssva1": {
        "autoencoder": {
            "checkpoint": "./checkpoints/autoencoder/ssva1/epoch_49.pth",
            "anomaly_score_file": "./results/autoencoder/ssva1/epoch_49_roc_scores.auc_0.90.csv",
        },
        "vae": {
            "checkpoint": "./checkpoints/vae/ssva1/epoch_45.pth",
            "anomaly_score_file": "./results/vae/ssva1/epoch_45_roc_scores.auc_0.87.csv",
        },
        "fanogan": {
            "checkpoint": "./checkpoints/fanogan/ssva1/epoch_39.pth",
            "anomaly_score_file": "./results/fanogan/ssva1/epoch_39_roc_scores.auc_0.84.csv",
        },
        "ganomaly": {
            "checkpoint": "./checkpoints/ganomaly/ssva1/epoch_98.pth",
            "anomaly_score_file": "./results/ganomaly/ssva1/epoch_98_roc_scores.auc_0.90.csv",
        },
        "lraad": {
            "checkpoint": "./checkpoints/lraad/ssva1/epoch_70.pth",
            "anomaly_score_file": "./results/lraad/ssva1/epoch_70_roc_recon_losses.auc_0.94.csv",
        },
        "improved_lraad": {
            "checkpoint": "./checkpoints/improved_lraad/ssva1/epoch_36.pth",
            "anomaly_score_file": "./results/improved_lraad/ssva1/epoch_36_roc_scores.auc_0.97.csv",
        },
    },
    "ssva2": {
        "autoencoder": {
            "checkpoint": "./checkpoints/autoencoder/ssva2/epoch_49.pth",
            "anomaly_score_file": "./results/autoencoder/ssva2/epoch_49_roc_scores.auc_0.89.csv",
        },
        "vae": {
            "checkpoint": "./checkpoints/vae/ssva2/epoch_46.pth",
            "anomaly_score_file": "./results/vae/ssva2/epoch_46_roc_scores.auc_0.87.csv",
        },
        "fanogan": {
            "checkpoint": "./checkpoints/fanogan/ssva2/epoch_15.pth",
            "anomaly_score_file": "./results/fanogan/ssva2/epoch_15_roc_scores.auc_0.86.csv",
        },
        "ganomaly": {
            "checkpoint": "./checkpoints/ganomaly/ssva2/epoch_98.pth",
            "anomaly_score_file": "./results/ganomaly/ssva2/epoch_98_roc_scores.auc_0.92.csv",
        },
        "lraad": {
            "checkpoint": "./checkpoints/lraad/ssva2/epoch_88.pth",
            "anomaly_score_file": "./results/lraad/ssva2/epoch_88_roc_recon_losses.auc_0.95.csv",
        },
        "improved_lraad": {
            "checkpoint": "./checkpoints/improved_lraad/ssva2/epoch_99.pth",
            "anomaly_score_file": "./results/improved_lraad/ssva2/epoch_99_roc_scores.auc_0.98.csv",
        },
    },
    "ssva3": {
        "autoencoder": {
            "checkpoint": "./checkpoints/autoencoder/ssva3/epoch_49.pth",
            "anomaly_score_file": "./results/autoencoder/ssva3/epoch_49_roc_scores.auc_0.93.csv",
        },
        "vae": {
            "checkpoint": "./checkpoints/vae/ssva3/epoch_39.pth",
            "anomaly_score_file": "./results/vae/ssva3/epoch_39_roc_scores.auc_0.90.csv",
        },
        "fanogan": {
            "checkpoint": "./checkpoints/fanogan/ssva3/epoch_58.pth",
            "anomaly_score_file": "./results/fanogan/ssva3/epoch_58_roc_scores.auc_0.87.csv",
        },
        "ganomaly": {
            "checkpoint": "./checkpoints/ganomaly/ssva3/epoch_99.pth",
            "anomaly_score_file": "./results/ganomaly/ssva3/epoch_99_roc_scores.auc_0.93.csv",
        },
        "lraad": {
            "checkpoint": "./checkpoints/lraad/ssva3/epoch_66.pth",
            "anomaly_score_file": "./results/lraad/ssva3/epoch_66_roc_recon_losses.auc_0.96.csv",
        },
        "improved_lraad": {
            "checkpoint": "./checkpoints/improved_lraad/ssva3/epoch_91.pth",
            "anomaly_score_file": "./results/improved_lraad/ssva3/epoch_91_roc_scores.auc_0.99.csv",
        },
    },
    "ssva4": {
        "autoencoder": {
            "checkpoint": "./checkpoints/autoencoder/ssva4/epoch_43.pth",
            "anomaly_score_file": "./results/autoencoder/ssva4/epoch_43_roc_scores.auc_0.91.csv",
        },
        "vae": {
            "checkpoint": "./checkpoints/vae/ssva4/epoch_36.pth",
            "anomaly_score_file": "./results/vae/ssva4/epoch_36_roc_scores.auc_0.89.csv",
        },
        "fanogan": {
            "checkpoint": "./checkpoints/fanogan/ssva4/epoch_45.pth",
            "anomaly_score_file": "./results/fanogan/ssva4/epoch_45_roc_scores.auc_0.87.csv",
        },
        "ganomaly": {
            "checkpoint": "./checkpoints/ganomaly/ssva4/epoch_99.pth",
            "anomaly_score_file": "./results/ganomaly/ssva4/epoch_99_roc_scores.auc_0.93.csv",
        },
        "lraad": {
            "checkpoint": "./checkpoints/lraad/ssva4/epoch_98.pth",
            "anomaly_score_file": "./results/lraad/ssva4/epoch_98_roc_recon_losses.auc_0.96.csv",
        },
        "improved_lraad": {
            "checkpoint": "./checkpoints/improved_lraad/ssva4/epoch_64.pth",
            "anomaly_score_file": "./results/improved_lraad/ssva4/epoch_64_roc_scores.auc_0.98.csv",
        },
    },
    "ssva5": {
        "autoencoder": {
            "checkpoint": "./checkpoints/autoencoder/ssva5/epoch_49.pth",
            "anomaly_score_file": "./results/autoencoder/ssva5/epoch_49_roc_scores.auc_0.90.csv",
        },
        "vae": {
            "checkpoint": "./checkpoints/vae/ssva5/epoch_43.pth",
            "anomaly_score_file": "./results/vae/ssva5/epoch_43_roc_scores.auc_0.90.csv",
        },
        "fanogan": {
            "checkpoint": "./checkpoints/fanogan/ssva5/epoch_48.pth",
            "anomaly_score_file": "./results/fanogan/ssva5/epoch_48_roc_scores.auc_0.89.csv",
        },
        "ganomaly": {
            "checkpoint": "./checkpoints/ganomaly/ssva5/epoch_99.pth",
            "anomaly_score_file": "./results/ganomaly/ssva5/epoch_99_roc_scores.auc_0.95.csv",
        },
        "lraad": {
            "checkpoint": "./checkpoints/lraad/ssva5/epoch_34.pth",
            "anomaly_score_file": "./results/lraad/ssva5/epoch_34_roc_recon_losses.auc_0.97.csv",
        },
        "improved_lraad": {
            "checkpoint": "./checkpoints/improved_lraad/ssva5/epoch_99.pth",
            "anomaly_score_file": "./results/improved_lraad/ssva5/epoch_99_roc_scores.auc_1.00.csv",
        },
    },
    "ssva6": {
        "autoencoder": {
            "checkpoint": "./checkpoints/autoencoder/ssva6/epoch_49.pth",
            "anomaly_score_file": "./results/autoencoder/ssva6/epoch_49_roc_scores.auc_0.85.csv",
        },
        "vae": {
            "checkpoint": "./checkpoints/vae/ssva6/epoch_39.pth",
            "anomaly_score_file": "./results/vae/ssva6/epoch_39_roc_scores.auc_0.82.csv",
        },
        "fanogan": {
            "checkpoint": "./checkpoints/fanogan/ssva6/epoch_36.pth",
            "anomaly_score_file": "./results/fanogan/ssva6/epoch_36_roc_scores.auc_0.81.csv",
        },
        "ganomaly": {
            "checkpoint": "./checkpoints/ganomaly/ssva6/epoch_5.pth",
            "anomaly_score_file": "./results/ganomaly/ssva6/epoch_5_roc_scores.auc_0.83.csv",
        },
        "lraad": {
            "checkpoint": "./checkpoints/lraad/ssva6/epoch_5.pth",
            "anomaly_score_file": "./results/lraad/ssva6/epoch_5_roc_recon_losses.auc_0.87.csv",
        },
        "improved_lraad": {
            "checkpoint": "./checkpoints/improved_lraad/ssva6/epoch_83.pth",
            "anomaly_score_file": "./results/improved_lraad/ssva6/epoch_83_roc_scores.auc_0.98.csv",
        },
    },
}


def calc_partial_auc(fpr, tpr, fpr_upper_limit=0.1):
    """
    Calculate partial AUC
    Args:
        fpr: np.array, false positive rate
        tpr: np.array, true positive rate
    Returns:
        partial_auc: float, partial AUC
    """
    partial_fpr = fpr[fpr <= fpr_upper_limit]
    partial_tpr = tpr[: len(partial_fpr)]
    partial_auc = auc(partial_fpr, partial_tpr) / fpr_upper_limit
    return partial_auc


class Evaluater:
    def __init__(self, config):
        self.config = config
        self.datasets = list(config.keys())
        self.models = [
            "autoencoder",
            "vae",
            "fanogan",
            "ganomaly",
            "lraad",
            "improved_lraad",
        ]
        self.auc_results = {dataset: {} for dataset in self.datasets}
        self.partial_auc_results_01 = {dataset: {} for dataset in self.datasets}
        self.partial_auc_results_02 = {dataset: {} for dataset in self.datasets}
        self.partial_auc_results_03 = {dataset: {} for dataset in self.datasets}

    def read_anomaly_score(self, anomaly_score_file) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read anomaly score from csv file
        Args:
            anomaly_score_file: str, path to csv file
        Returns:
            anomaly_score: np.array, anomaly score
            label: np.array, label
        """
        df = pd.read_csv(anomaly_score_file)
        scores = np.array(df["score"].values)
        labels = np.array(df["label"].values)
        return scores, labels

    def evaluate(self, evals, save_root="experiments"):
        for model in self.models:
            model_config = evals[model]
            checkpoint = model_config["checkpoint"]
            args = train_args[model]
            command = [
                "python",
                f"./src/models/{model}/train.py",
                f"--dataset {self.current_dataset}",
                f"--pretrained {checkpoint}",
                "--eval",
                f"--eval_save {save_root}",
            ] + args
            subprocess.call(" ".join(command), shell=True)

    def plot_roc_curve(self, evals, save_path=None, dataset_name=""):
        """
        Plot ROC curve
        Args:
            evals: list, list of evaluation config
        """
        plt.figure()
        # make figure beautiful
        for model in self.models:
            model_config = evals[model]
            # read anomaly score
            anomaly_score, label = self.read_anomaly_score(
                model_config["anomaly_score_file"]
            )

            anomaly_score = -anomaly_score
            label = 1 - label

            # calculate roc curve
            fpr, tpr, _ = roc_curve(label, anomaly_score, pos_label=1)
            plt.plot(fpr, tpr, label=model)

            # calculate AUC
            auc_score = auc(fpr, tpr)
            self.auc_results[self.current_dataset][model] = auc_score
            partial_auc_01 = calc_partial_auc(fpr, tpr, fpr_upper_limit=0.1)
            self.partial_auc_results_01[self.current_dataset][model] = partial_auc_01
            partial_auc_02 = calc_partial_auc(fpr, tpr, fpr_upper_limit=0.2)
            self.partial_auc_results_02[self.current_dataset][model] = partial_auc_02
            partial_auc_03 = calc_partial_auc(fpr, tpr, fpr_upper_limit=0.3)
            self.partial_auc_results_03[self.current_dataset][model] = partial_auc_03

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({dataset_name})")
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def run(self, save_root="experiments"):
        for dataset in self.datasets:
            if dataset == "ssva1":
                dataset_name = "Dataset 1"
            elif dataset == "ssva2":
                dataset_name = "Dataset 2"
            elif dataset == "ssva3":
                dataset_name = "Dataset 3"
            elif dataset == "ssva4":
                dataset_name = "Dataset 4"
            elif dataset == "ssva5":
                dataset_name = "Dataset 5"
            elif dataset == "ssva6":
                dataset_name = "Dataset 6"
            else:
                raise ValueError("Invalid dataset name")
            print(f"Evaluating {dataset_name}")
            self.current_dataset = dataset
            evals = self.config[dataset]
            save_path = Path(save_root) / dataset
            save_path.mkdir(parents=True, exist_ok=True)

            self.plot_roc_curve(
                evals, save_path=save_path / "roc_curve.pdf", dataset_name=dataset_name
            )
            self.evaluate(evals, save_root="./experiments/")

        # save results
        # auc
        with open(Path(save_root) / "auc_results.json", "w") as f:
            json.dump(self.auc_results, f, indent=4)
        # save to csv using pandas, retain 4 decimal places
        df = pd.DataFrame(self.auc_results).T
        df.to_csv(Path(save_root) / "auc_results.csv", float_format="%.4f")
        # partial auc
        with open(Path(save_root) / "partial_auc_results_01.json", "w") as f:
            json.dump(self.partial_auc_results_01, f, indent=4)
        with open(Path(save_root) / "partial_auc_results_02.json", "w") as f:
            json.dump(self.partial_auc_results_02, f, indent=4)
        with open(Path(save_root) / "partial_auc_results_03.json", "w") as f:
            json.dump(self.partial_auc_results_03, f, indent=4)
        df = pd.DataFrame(self.partial_auc_results_01).T
        df.to_csv(Path(save_root) / "partial_auc_results_01.csv", float_format="%.4f")
        df = pd.DataFrame(self.partial_auc_results_02).T
        df.to_csv(Path(save_root) / "partial_auc_results_02.csv", float_format="%.4f")
        df = pd.DataFrame(self.partial_auc_results_03).T
        df.to_csv(Path(save_root) / "partial_auc_results_03.csv", float_format="%.4f")


if __name__ == "__main__":
    evaluater = Evaluater(eval_config)
    evaluater.run()
