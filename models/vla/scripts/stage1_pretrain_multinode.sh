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


echo "stage1: multimodel pretraining"


NNODES=4
GPU_PER_NODES=8
PRIORITY=NORMAL # HIGHEST
EXP=pretrain7B_motion_audio_final

IMG=registry.st-sh-01.sensecore.cn/XXXXX
PARTITION_ID=YYYYYYYYYYYYYY
# PARTITION_ID=vigen-2
WORKSPACE_ID=YYYYYYYYYYYYYY
GPUS=N1lS.Ia.I20.8,N1lS.Ia.I20.4,N1lS.Ia.I20.2,N1lS.Ia.I20.1
GPUS_=""
for((i=0,j=0;i<=${#GPUS};++i)); do
	if [[ ${i} -ge ${#GPUS} ]] \
	|| [[ ${GPUS:i:1} == "," ]]; then
		GPUS_=${GPUS_}" "${GPUS:j:i-j}
		j=$((${i}+1))
	fi
done
GPUS=($(echo ${GPUS_}))
DATETIME=$(date '+%Y-%m-%d-%H:%M:%S')
if [ ${GPU_PER_NODES} -ge 8 ]; then
	GPU=${GPUS[0]}
elif [ ${GPU_PER_NODES} -ge 4 ]; then
	GPU=${GPUS[1]}
elif [ ${GPU_PER_NODES} -ge 2 ]; then
	GPU=${GPUS[2]}
else
	GPU=${GPUS[3]}
fi
NUM_GPU=$((${GPU_PER_NODES}*${NNODES}))
# HOME_DIR=${0%/*}
echo "HOME_DIR: ${HOME_DIR}"
HOME_DIR="/root/pengyang/codebase/SOLAMI/models/vla"
cd ${HOME_DIR}
HOME_DIR=$(pwd)
echo "HOME_DIR: ${HOME_DIR}"

CMD=" /root/pengyang/codebase/SOLAMI/models/vla/anygpt/train/audio_motion_pretrain.py \
    --deepspeed "/root/pengyang/codebase/SOLAMI/models/vla/scripts/stage1_deepspeed_zero3.json" \
    --run_name "audio_motion_pretrain_final" \
    --model_name_or_path '${METAROOT}' \
    --speech_data_path '${speech_datasets}' \
    --motion_data_path '${motion_datasets}' \
    --it_data_path '${it_datasets}' \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 False \
    --fp16 True \
    --do_train \
    --do_eval \
    --output_dir "${OUTROOT}" \
    --model_max_length 1024 \
    --block_size 1024 \
    --save_strategy "steps" \
    --save_steps 512 \
    --evaluation_strategy "steps" \
    --eval_steps 512 \
    --max_steps 4097 \
    --num_train_epochs 1 \
    --val_set_size 1024 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \
    --overwrite_output_dir True\
    --use_flash_attn False \
    --save_total_limit 10 \
    --concatenating True" 
    # --report_to 'wandb' "

# if [ ${NNODES} -gt 1 ]; then
echo "CMD: ${CMD}"
srun --partition-id ${PARTITION_ID} \
    --workspace-id ${WORKSPACE_ID} \
    --framework pt \
    --job-name ${EXP} \
    --resource ${GPU} \
    --distributed AllReduce \
    --output outputs/run_${DATETIME}.log \
    --nodes ${NNODES} \
    --priority ${PRIORITY} \
    --container-image ${IMG} \
     bash -c "conda activate Final && export HOME='/mnt/AFS_jiangjianping' && cd "${HOME_DIR}"; python -m torch.distributed.run --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT --nproc-per-node "${GPU_PER_NODES}" --nnodes \$WORLD_SIZE --node_rank \$RANK ${CMD}"
