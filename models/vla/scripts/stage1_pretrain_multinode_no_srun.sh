#!/bin/bash
# export NCCL_DEBUG=INFO
# export NCCL_IB_GID_INDEX=3

# METAROOT="Path-to-pretrained-model" 
# https://huggingface.co/fnlp/AnyGPT-base
METAROOT="/root/pengyang/codebase/SOLAMI/extra/AnyGPT-base" 
# METAROOT="/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain/checkpoint-10"
DATAROOT="SOLAMI_data"

OUTROOT="/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain_audio_motion_final"
CACHEROOT="/root/pengyang/codebase/SOLAMI/models/vla/data/pretrain/cache"


speech_datasets="${DATAROOT}/audio/commonvoice_processed/commonvoice_merged.jsonl ${DATAROOT}/audio/anyinstruct/anyinstruct_merged.jsonl"
motion_datasets="${DATAROOT}/tmp_data/pretrain_new_tokens/local_cont6d_body_hand_sep/motion_train_merged.jsonl ${DATAROOT}/tmp_data/pretrain_new_tokens/local_cont6d_body_hand_sep/motion_test_merged.jsonl"
it_datasets="${DATAROOT}/Conversation/train_it_items.jsonl"



# python -m torch.distributed.run  --nproc-per-node 8 \
python  /root/pengyang/codebase/SOLAMI/models/vla/anygpt/train/audio_motion_pretrain.py \
    --deepspeed "/root/pengyang/codebase/SOLAMI/models/vla/scripts/stage1_deepspeed.json" \
    --run_name "audio_motion_pretrain_final" \
    --model_name_or_path ${METAROOT} \
    --speech_data_path "${speech_datasets}" \
    --motion_data_path "${motion_datasets}" \
    --it_data_path "${it_datasets}" \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 False \
    --fp16 True \
    --do_train \
    --do_eval \
    --output_dir "${OUTROOT}" \
    --model_max_length 1024 \
    --block_size 1024 \
    --save_strategy "steps" \
    --save_steps 500 \
    --evaluation_strategy "steps" \
    --eval_steps 16 \
    --max_steps 4096 \
    --num_train_epochs 1 \
    --val_set_size 128 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \
    --overwrite_output_dir True\
    --use_flash_attn False \
    --save_total_limit 8 \
    --concatenating True


