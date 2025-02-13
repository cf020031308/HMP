#!/bin/bash -
gpu=0
fn=$(basename "$0")
classes=5
hidden=64
args="--heads 1 --AllSet_num_heads 1 --hidden $hidden --MLP_hidden $hidden --Classifier_hidden $hidden --epochs 1000 --patience 1000 --display_step -1 --raw_data_dir ./raw_data/chain --runs 10 --dropout 0 --input_dropout 0"
for method in AllSetTransformer EDGNN UniGCNII AllDeepSets NT2 HyperGCN LEGCN; do
    for length in $(seq 1 10); do
        noise=0
        width=1
        echo "method: ${method}, length: ${length}, width: ${width}, noise: ${noise}, hidden: ${hidden}"
        CUDA_VISIBLE_DEVICES=$gpu python3 -u train.py --method $method --dname "chain-${width}-${length}-${classes}" --data_dir "./data/chain-${width}-${length}-${classes}" --n-layers "$(( 2 * $length - 1))" --All_num_layers $length $args --feature_noise $noise
        noise=1
        for width in 1 2 3; do
            echo "method: ${method}, length: ${length}, width: ${width}, noise: ${noise}, hidden: ${hidden}"
            CUDA_VISIBLE_DEVICES=$gpu python3 -u train.py --method $method --dname "chain-${width}-${length}-${classes}" --data_dir "./data/chain-${width}-${length}-${classes}" --n-layers "$(( 2 * $length - 1))" --All_num_layers $length $args --feature_noise $noise
        done
    done
done 2>&- | tee -a logs/$fn.log
