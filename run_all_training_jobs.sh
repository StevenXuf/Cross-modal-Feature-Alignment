#!/bin/bash

datasets=("coco_train")
network="transformer"
loss_func="mse"
gpu_id=0

tasks=("img2txt" "txt2img")
models=("clip" "blip_official_itm" "metatransformer" "resnetbert" "convitroberta" "convnextxlnet")

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for dataset in "${datasets[@]}"; do

            logfile="${network}_${model}_${dataset}_${task}_${loss_func}.log"

            python transformer_train.py \
                "$model" \
                "$dataset" \
                --network $network \
                --loss_func $loss_func \
                --task "$task" \
                --gpu_id $gpu_id \
                &> "$logfile" &

        done
    done
done

wait

echo "All jobs finished."