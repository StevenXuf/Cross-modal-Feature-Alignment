#!/bin/sh -l
cd
source ./my-env/bin/activate

model_names='clip'
datasets='coco_train'
n_epochs=100
batch_size=64
img_retrieval='yes'
loss_func='mse'
hidden_sizes=(500 300 100)

for model in $model_names
    do
        for dataset in $datasets
        do
                torchrun \
                --standalone \
                --nnodes=1 \
                --nproc-per-node=3 \
                /home/fxu/alignment/train.py \
                $model \
                $dataset \
                ${hidden_sizes[@]} \
                --n_epochs=$n_epochs \
                --batch_size=$batch_size \
                --img_retrieval=$img_retrieval \
                --loss_func=$loss_func
        done
done

echo "*****************************************"
echo "training ${model} for ${dataset} with ${loss_func} finished!"
echo "*****************************************"
