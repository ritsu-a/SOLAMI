#!/bin/bash
# export NCCL_DEBUG=INFO
# export NCCL_IB_GID_INDEX=3

# path-to-pretrain-model
METAROOT="/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain_audio_motion_final/checkpoint-2560" 

DATAROOT="SOLAMI_data"
OUTROOT="/root/pengyang/codebase/SOLAMI/models/vla/output_models/it_lora_deepspeed"
CACHEROOT="${DATAROOT}/cache/it_debug"

it_datasets="${DATAROOT}/Conversation/train_it_items.jsonl"


echo "stage2: instruction tuning"

# export CUDA_VISIBLE_DEVICES=0,

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun \
    --nproc_per_node 8 \
     /root/pengyang/codebase/SOLAMI/models/vla/anygpt/train/conv_instruction_tuning.py \
     --deepspeed "/root/pengyang/codebase/SOLAMI/models/vla/scripts/stage2_deepspeed_offload.json" \
    --model_name_or_path "${METAROOT}" \
    --run_name "mm_sft" \
    --it_data_path "${it_datasets}" \
    --cache_dir ${CACHEROOT} \
    --it_type 'lora deepspeed' \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules 'q_proj,v_proj' \
    --preprocessing_num_workers 4 \
    --bf16 False \
    --fp16 True \
    --do_train \
    --do_eval \
    --output_dir "${OUTROOT}" \
    --model_max_length 4096 \
    --block_size 4096 \
    --save_strategy "steps" \
    --save_only_model \
    --save_steps 128 \
    --evaluation_strategy "steps" \
    --eval_steps 64 \
    --max_steps 1290 \
    --concatenating True \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 6 \
    --val_set_size 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \
    --use_flash_attn False \
    --overwrite_output_dir True\
    --save_total_limit 15 \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --use_flash_attn True \
    # --ddp_timeout 7200 \
