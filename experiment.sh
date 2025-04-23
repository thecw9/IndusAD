#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

model=$1
script="src/models/$model/train.py"
echo "Running $model"

if [ "$model" == "help" ]; then
	echo "Usage: ./experiment_lraad.sh <model>"
	echo "Models:"
	echo "  lraad: LRAAD"
	echo "  improved_lraad: Improved LRAAD"
	echo "  autoencoder: Autoencoder"
	echo "  vae: Variational Autoencoder"
	echo "  fanogan: F-AnoGAN"
	echo "  ganomaly: GANomaly"
	exit 0
fi

if [ "$model" == "lraad" ]; then
	common_args="--num_epochs 100 \
  --beta_kl 1 \
  --num_vae_iter 10 \
  --final_activation none \
  --z_dim 32 \
  --threshold_low 100 \
  --threshold_high 100 \
  --decay 0.0001 \
  --beta_adv 1"
	datasets=("ssva1" "ssva2" "ssva3" "ssva4" "ssva5" "ssva6")
	seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
elif [ "$model" == "improved_lraad" ]; then
	common_args="--num_epochs 100 \
  --beta_kl 1"
	datasets=("ssva1" "ssva2" "ssva3" "ssva4" "ssva5" "ssva6")
	seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
elif [ "$model" == "autoencoder" ]; then
	common_args="--num_epochs 50 \
  --recon_loss_type mse"
	datasets=("ssva1" "ssva2" "ssva3" "ssva4" "ssva5" "ssva6")
	seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
elif [ "$model" == "vae" ]; then
	common_args="--num_epochs 50 \
  --beta_kl 1 \
  --z_dim 32"
	datasets=("ssva1" "ssva2" "ssva3" "ssva4" "ssva5" "ssva6")
	seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
elif [ "$model" == "fanogan" ]; then
	common_args="--num_epochs 100"
	datasets=("ssva1" "ssva2" "ssva3" "ssva4" "ssva5" "ssva6")
	seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
elif [ "$model" == "ganomaly" ]; then
	common_args="--num_epochs 100"
	datasets=("ssva1" "ssva2" "ssva3" "ssva4" "ssva5" "ssva6")
	seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
else
	echo "Model not found"
	exit 1
fi

for i in "${!datasets[@]}"; do
	dataset=${datasets[$i]}
	seed=${seeds[$i]}
	if [ ! -d "results/$model/$dataset" ]; then
		python $script $common_args --dataset $dataset --seed $seed
	else
		echo "results/$model/$dataset already exists"
	fi
done
