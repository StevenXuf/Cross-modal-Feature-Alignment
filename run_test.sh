#!/bin/sh -l
cd
source ./my-env/bin/activate

model_names='clip' # blip metatransformer resnetbert convitroberta convnextxlnet'
datasets='imdb'
loss_func='mse'
checkpoint=17
hidden_size=(500 300 100)

for model_name in $model_names
do
        for dataset in $datasets
        do
		python3 \
		/home/users/u101139/alignment/test.py \
		$model_name \
		$dataset \
		$checkpoint 
		
		echo "*****************************************"
                echo "test ${model_name} for ${dataset} finished!"
                echo "*****************************************"
        done
done
