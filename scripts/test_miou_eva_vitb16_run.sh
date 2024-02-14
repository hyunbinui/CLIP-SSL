#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2             # num process per node
#SBATCH --cpus-per-task=1               # num cpu cores per process (total num core of a node / total num gpus of a node * requested num gpu)
#SBATCH --wait-all-nodes=1
#SBATCH --mem=50000MB                   # Using 10GB CPU Memory (MIN_MEMORY)
#SBATCH --job-name=basic_test_miou
#SBATCH --time=12:00:00
#SBATCH --output=./src/slurm_logs/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate openclip

export MASTER_PORT=12801

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

cd src

srun --cpu-bind=v --accel-bind=gn python -m training.main --save-most-recent --delete-previous-checkpoint --logs /shared/s2/lab01/jiwoosong/logs  \
--batch-size=1 --miou_frequency=1  \
--model EVA02-CLIP-B-16 --pretrained eva --test-type coco_panoptic --train-data="" \
--val-data /shared/s2/lab01/dataset/zeroseg/coco/annotations/panoptic_val2017_before.json \
--embed-path ../metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy \
--val-image-root /shared/s2/lab01/dataset/zeroseg/coco/val2017 --cache-dir ../checkpoints/EVA02_CLIP_B_psz16_s8B.pt --extract-type="v2" \
--name test_coco_panoptic_eva_vitb16_miou_$current_time --downsample-factor 16 --det-image-size 1024