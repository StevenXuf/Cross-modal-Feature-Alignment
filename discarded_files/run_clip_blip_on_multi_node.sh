#!/bin/sh -l
#SBATCH --time=4:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=32
#SBATCH --account=p200248

export MASTER_PORT=12345

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

module load Python/3.10.4-GCCcore-11.3.0
cd /home/users/u101139
source ./my-env/bin/activate
Temp='openai/clip-vit-base-patch32'

checkpoints='Salesforce/blip-image-captioning-base'
which_features='val'

for checkpoint in $checkpoints
do
	if [ $checkpoint = 'Salesforce/blip-image-captioning-base' ];
	then
		batch_size=512
	else
		batch_size=2048
	fi	
        for which in $which_features
        do
                       srun torchrun \
				--nnodes=2 \
				--nproc-per-node=4 \
				--rdzv-id=001 \
				--rdzv-backend=c10d \
				--rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
				/home/users/u101139/alignment/ClipBlipExtractor_multi_node.py \
				$checkpoint \
				--which_features=$which \
				--batch_size=$batch_size

			echo "*****************************************"
                        echo "${checkpoint} for ${which} finished!"
                        echo "*****************************************"
        done
done
