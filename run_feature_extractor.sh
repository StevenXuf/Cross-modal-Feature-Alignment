#!/bin/sh -l
cd
source ./my-env/bin/activate

model_names='blip'
datasets='imdb'
batch_size=512

for model_name in $model_names
do
	for dataset in $datasets
	do
		torchrun \
			--nnodes=2 \
			--nproc-per-node=4 \
			/home/fxu/alignment/feature_extractor.py \
			$model_name \
			$dataset \
			--batch_size=$batch_size
		echo "*****************************************"
        echo "${model_name} ${dataset} extraction finished!"
        echo "*****************************************"
	done
done
