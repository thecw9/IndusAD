from pathlib import Path
import subprocess
import os
import sys
import json
import numpy as np
from typing import Tuple
from scipy.sparse import data
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
)
import scienceplots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("husl", 6)  # 使用 "husl" 色板生成 5 种颜色
colors = ["#88ccee", "#44aa99", "#117733", "#999933", "#ddcc77", "#aa4499"]
colors = ["#edc948", "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
colors = ["#edc948", "#4eaaff", "#f28e2b", "#e15759", "#76b7b2", "#aa4499"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#4eaaff", "#8c564b"]

os.environ["PYTHONPATH"] = os.getcwd()

# plt.style.use(["science", "nature"])
# plt.rcParams.update({"font.size": 20})
# plt.rcParams["lines.linewidth"] = 1.5

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False

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
            "checkpoint": "./checkpoints/autoencoder/ssva2/epoch_28.pth",
            "anomaly_score_file": "./results/autoencoder/ssva2/epoch_28_roc_scores.auc_0.89.csv",
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
            "checkpoint": "./checkpoints/autoencoder/ssva3/epoch_46.pth",
            "anomaly_score_file": "./results/autoencoder/ssva3/epoch_46_roc_scores.auc_0.93.csv",
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
            "checkpoint": "./checkpoints/autoencoder/ssva4/epoch_31.pth",
            "anomaly_score_file": "./results/autoencoder/ssva4/epoch_31_roc_scores.auc_0.91.csv",
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
            "checkpoint": "./checkpoints/autoencoder/ssva5/epoch_28.pth",
            "anomaly_score_file": "./results/autoencoder/ssva5/epoch_28_roc_scores.auc_0.90.csv",
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
            "checkpoint": "./checkpoints/autoencoder/ssva6/epoch_46.pth",
            "anomaly_score_file": "./results/autoencoder/ssva6/epoch_46_roc_scores.auc_0.85.csv",
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
        # "lraad": {
        #     "checkpoint": "./checkpoints/lraad/ssva6/epoch_20.pth",
        #     "anomaly_score_file": "./results/lraad/ssva6/epoch_20_roc_recon_losses.auc_0.86.csv",
        # },
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


def find_best_threshold(fpr, tpr, thresholds):
    """
    Find best threshold
    Args:
        fpr: np.array, false positive rate
        tpr: np.array, true positive rate
        thresholds: np.array, thresholds
    Returns:
        best_threshold: float, best threshold
    """
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return best_threshold


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
            # "improved_lraad",
        ]
        self.label = {
            "autoencoder": "CAE",
            "vae": "VAE",
            "fanogan": "f-AnoGAN",
            "ganomaly": "GANomaly",
            "lraad": "LRAAD",
            "improved_lraad": "Soft-LRAAD",
        }
        self.auc_results = {dataset: {} for dataset in self.datasets}
        self.recall_results = {dataset: {} for dataset in self.datasets}
        self.accuracy_results = {dataset: {} for dataset in self.datasets}
        self.f1_results = {dataset: {} for dataset in self.datasets}
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

    def plot_roc_curve(self, evals, save_path, dataset_name=""):
        """
        Plot ROC curve
        Args:
            evals: list, list of evaluation config
        """
        plt.figure()
        # make figure beautiful
        for i, model in enumerate(self.models):
            model_config = evals[model]
            # read anomaly score
            anomaly_score, label = self.read_anomaly_score(
                model_config["anomaly_score_file"]
            )

            anomaly_score = -anomaly_score
            label = 1 - label

            # calculate roc curve
            fpr, tpr, thresholds = roc_curve(label, anomaly_score, pos_label=1)
            plt.plot(fpr, tpr, color=colors[i], label=self.label[model])

        plt.xlabel("假阳率（FPR）")
        plt.ylabel("真阳率（TPR）")
        plt.title(f"ROC曲线（{dataset_name}）")
        plt.legend()
        # make dir
        os.makedirs(Path(save_path).parent, exist_ok=True)
        if save_path is not None:
            plt.savefig(save_path, dpi=1200)
            # save with svg
            plt.savefig(Path(save_path).with_suffix(".svg"))
            plt.savefig(Path(save_path).with_suffix(".pdf"))
        else:
            plt.show()
        plt.close()

    def calc_partial_auc(
        self,
        evals,
        save_root="experiments/",
        dataset="",
    ):
        """
        Plot ROC curve
        Args:
            evals: list, list of evaluation config
        """
        plt.figure()
        # make figure beautiful
        for i, model in enumerate(self.models):
            model_config = evals[model]
            # read anomaly score
            anomaly_score, label = self.read_anomaly_score(
                model_config["anomaly_score_file"]
            )

            anomaly_score = -anomaly_score
            label = 1 - label

            # calculate roc curve
            fpr, tpr, thresholds = roc_curve(label, anomaly_score, pos_label=1)
            plt.plot(fpr, tpr, color=colors[i], label=self.label[model])

            # calculate AUC
            auc_score = roc_auc_score(label, anomaly_score)
            self.auc_results[dataset][model] = auc_score
            partial_auc_01 = roc_auc_score(label, anomaly_score, max_fpr=0.1)
            self.partial_auc_results_01[dataset][model] = partial_auc_01
            partial_auc_02 = roc_auc_score(label, anomaly_score, max_fpr=0.2)
            self.partial_auc_results_02[dataset][model] = partial_auc_02
            partial_auc_03 = roc_auc_score(label, anomaly_score, max_fpr=0.3)
            self.partial_auc_results_03[dataset][model] = partial_auc_03
            # auc_score = auc(fpr, tpr)
            # self.auc_results[dataset][model] = auc_score
            # partial_auc_01 = calc_partial_auc(fpr, tpr, fpr_upper_limit=0.1)
            # self.partial_auc_results_01[dataset][model] = partial_auc_01
            # partial_auc_02 = calc_partial_auc(fpr, tpr, fpr_upper_limit=0.2)
            # self.partial_auc_results_02[dataset][model] = partial_auc_02
            # partial_auc_03 = calc_partial_auc(fpr, tpr, fpr_upper_limit=0.3)
            # self.partial_auc_results_03[dataset][model] = partial_auc_03

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

    def evaluate_classify(self, evals, save_root, dataset="", dataset_name=""):
        """
        Evaluate classification
        Args:
            evals: list, list of evaluation config
        """
        os.makedirs(save_root, exist_ok=True)
        plt.figure()
        # make figure beautiful
        for i, model in enumerate(self.models):
            model_config = evals[model]
            # read anomaly score
            anomaly_score, label = self.read_anomaly_score(
                model_config["anomaly_score_file"]
            )

            anomaly_score = -anomaly_score
            label = 1 - label

            fpr, tpr, thresholds = roc_curve(label, anomaly_score, pos_label=1)
            threshold = find_best_threshold(fpr, tpr, thresholds)
            y_pred = (anomaly_score > threshold).astype(int)
            cm = confusion_matrix(label, y_pred)

            plt.figure(figsize=(6, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["正常", "异常"],
                yticklabels=["正常", "异常"],
            )
            plt.xlabel("预测值")
            plt.ylabel("真实标签")
            plt.title(f"{self.label[model]} 混淆矩阵")
            plt.savefig(Path(save_root) / f"{model}_confusion_matrix.png", dpi=1200)
            plt.savefig(Path(save_root) / f"{model}_confusion_matrix.svg")
            plt.close()

            # PR curve
            precision, recall, _ = precision_recall_curve(label, anomaly_score)
            plt.plot(recall, precision, color=colors[i], label=self.label[model])

            # save accuracy, recall, f1_score
            self.accuracy_results[dataset][model] = accuracy_score(label, y_pred)
            self.recall_results[dataset][model] = recall_score(label, y_pred)
            self.f1_results[dataset][model] = f1_score(label, y_pred)




        plt.xlabel("召回率（Recall）")
        plt.ylabel("精确率（Precision）")
        plt.title(f"PR曲线（{dataset_name}）")
        plt.legend()
        plt.savefig(Path(save_root) / "pr_curve.png", dpi=1200)
        plt.savefig(Path(save_root) / "pr_curve.svg")
        plt.close()

        save_root = Path(save_root).parent
        with open(Path(save_root) / "accuracy_results.json", "w") as f:
            json.dump(self.accuracy_results, f, indent=4)
        df = pd.DataFrame(self.accuracy_results).T
        df.to_csv(Path(save_root) / "accuracy_results.csv", float_format="%.4f")

        with open(Path(save_root) / "recall_results.json", "w") as f:
            json.dump(self.recall_results, f, indent=4)
        df = pd.DataFrame(self.recall_results).T
        df.to_csv(Path(save_root) / "recall_results.csv", float_format="%.4f")


        with open(Path(save_root) / "f1_results.json", "w") as f:
            json.dump(self.f1_results, f, indent=4)
        df = pd.DataFrame(self.f1_results).T
        df.to_csv(Path(save_root) / "f1_results.csv", float_format="%.4f")

    def run(self, save_root="experiments"):
        for dataset in self.datasets:
            if dataset == "ssva1":
                dataset_name = "数据集 1"
            elif dataset == "ssva2":
                dataset_name = "数据集 2"
            elif dataset == "ssva3":
                dataset_name = "数据集 3"
            elif dataset == "ssva4":
                dataset_name = "数据集 4"
            elif dataset == "ssva5":
                dataset_name = "数据集 5"
            elif dataset == "ssva6":
                dataset_name = "数据集 6"
            else:
                raise ValueError("Invalid dataset name")
            print(f"Evaluating {dataset_name}")
            self.current_dataset = dataset
            evals = self.config[dataset]

            self.plot_roc_curve(
                evals,
                save_path=Path(save_root) / f"{dataset}_roc_curve.png",
                dataset_name=dataset_name,
            )
            self.calc_partial_auc(
                evals, save_root=save_root, dataset=self.current_dataset
            )
            self.evaluate_classify(
                evals,
                save_root=Path(save_root) / dataset,
                dataset=dataset,
                dataset_name=dataset_name,
            )
            self.evaluate(evals, save_root="./experiments/")


if __name__ == "__main__":
    evaluater = Evaluater(eval_config)
    evaluater.run()
