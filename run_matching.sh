#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=64
#SBATCH --account=p200248

models='convnextxlnet'
datasets='imdb flickr30k coco_train celebA'
metric='cosine'

module load Python/3.10.4-GCCcore-11.3.0
cd /home/users/u101139
source ./my-env/bin/activate

for model in $models
do
        for dataset in $datasets
        do
                echo "$model $dataset"
                python3 /home/users/u101139/alignment/match_acc_on_distance.py $model $dataset $metric
        done
done
