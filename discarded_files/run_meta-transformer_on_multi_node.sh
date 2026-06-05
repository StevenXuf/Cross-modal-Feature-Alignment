#!/bin/sh -l
#SBATCH --time=4:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=64
#SBATCH --account=p200248

export MASTER_PORT=12345

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

module load Python/3.10.4-GCCcore-11.3.0
cd /home/users/u101139
source ./my-env/bin/activate

which_datasets='imdb'
batch_size=1024

for which in $which_datasets
do
	srun torchrun \
		--nnodes=2 \
		--nproc-per-node=4 \
		--rdzv-id=001 \
		--rdzv-backend=c10d \
		--rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
		/home/users/u101139/alignment/meta-transformer_extractor_multi_node.py \
		--which_dataset=$which \
		--batch_size=$batch_size
	echo "*****************************************"
        echo "${which} finished!"
        echo "*****************************************"
done
