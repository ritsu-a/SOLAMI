#!/bin/bash
DATA_DIR=(
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_lora_checkpoint-768-final-0"
)
for i in "${!DATA_DIR[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python conv_evaluation.py --save_gt True --data_dir ${DATA_DIR[$i]} &
    wait
done
