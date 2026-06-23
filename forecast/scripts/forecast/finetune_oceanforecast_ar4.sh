#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH -p qgpu_a800
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=./slurmlogs/finetune-oceanforecast-autoregressive4-%j.out

# srun python src/train.py trainer=fsdp trainer.max_epochs=100 trainer.num_nodes=1 trainer.devices=2 datamodule=h5forecast_pretrain datamodule.batch_size=16 model=fourcastnet_pretrain model.after_scheduler.T_max=90 trainer.accumulate_grad_batches=1 hydra=hpc task_name=forecastnet_pretrain

srun python src/train.py trainer=gpu trainer.max_epochs=50 datamodule=oceanforecast datamodule.batch_size=8 datamodule.noise=0 datamodule.iter_num=4 model=oceanforecast_ar4 trainer.accumulate_grad_batches=1 hydra=hpc task_name=oceanforecast_autoregressive4
