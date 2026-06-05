#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=8
#SBATCH --account=p200248

module load Python/3.10.4-GCCcore-11.3.0-bare
cd /home/users/u101139
source ./my-env/bin/activate

checkpoints='openai/clip-vit-base-patch32 Salesforce/blip-image-captioning-base'
which_features='val train'

for checkpoint in $checkpoints
do
	for which in $which_features
	do
		       	srun python3 /home/users/u101139/load_data/ClipBlipExtractor_mono_node.py \
		       	$checkpoint \
			--n_gpus=4 \
		       	--which_features=$which \
		       	--batch_size=512
			
			echo "*****************************************"
        		echo "${which} finished!"
        		echo "*****************************************"
	done
done
