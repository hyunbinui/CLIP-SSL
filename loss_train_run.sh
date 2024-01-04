#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1             # num process per node
#SBATCH --cpus-per-task=1               # num cpu cores per process (total num core of a node / total num gpus of a node * requested num gpu)
#SBATCH --wait-all-nodes=1
#SBATCH --mem=10000MB                   # Using 10GB CPU Memory (MIN_MEMORY)
#SBATCH --job-name=cl_n1_g1_b2
#SBATCH --exclude=b[18,14,28]
#SBATCH --time=0-12:00:00
#SBATCH --output=./src/slurm_logs/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate openclip

export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

cd src

srun --cpu-bind=v --accel-bind=gn python -m training.main --use-contrastive-loss --save-most-recent --delete-previous-checkpoint  \
--batch-size=4 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model EVA02-CLIP-B-16 --pretrained eva --warmup 1000  --zeroshot-frequency 1 --dataset-type grid_distill  \
--test-type coco_panoptic --train-data /shared/s2/lab01/dataset/zeroseg/coco/annotations/instances_train2017.json \
--val-data /shared/s2/lab01/dataset/zeroseg/coco/annotations/panoptic_val2017.json \
--embed-path ../metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy --train-image-root /shared/s2/lab01/dataset/zeroseg/coco/train2017 \
--val-image-root /shared/s2/lab01/dataset/zeroseg/coco/val2017  --cache-dir ../checkpoints/EVA02_CLIP_B_psz16_s8B.pt --log-every-n-steps 50 \
--lock-image --save-frequency 6 --lock-image-unlocked-groups 12 --extract-type="v2" \
--name cl_coco_6_test1_eva_vitb16_12layers_$current_time --downsample-factor 16 --det-image-size 1024 \
--alpha 0.7
