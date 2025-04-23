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
        "improved_lraad": {
            "checkpoint": "./checkpoints/improved_lraad/ssva6/epoch_83.pth",
            "anomaly_score_file": "./results/improved_lraad/ssva6/epoch_83_roc_scores.auc_0.98.csv",
        },
    },
}


# collection all checkpoints to a list
checkpoints = []
for ssva, models in eval_config.items():
    for model, config in models.items():
        checkpoints.append(Path(config["checkpoint"]))

print(checkpoints)

# delete all files exclude checkpoints
for file in Path("./checkpoints").rglob("*.pth"):
    if file not in checkpoints:
        print(f"Delete {file}")
        file.unlink()
