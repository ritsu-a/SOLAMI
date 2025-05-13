#!/bin/bash
# export NCCL_DEBUG=INFO
# export NCCL_IB_GID_INDEX=3

# path-to-pretrain-model
METAROOT="/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain_audio_motion_final/checkpoint-2560" 

DATAROOT="SOLAMI_data"
OUTROOT="/root/pengyang/codebase/SOLAMI/models/vla/output_models/it_full_deepspeed_multinode"
CACHEROOT="${DATAROOT}/cache/it_debug"

it_datasets="${DATAROOT}/Conversation/train_it_items.jsonl ${DATAROOT}/Conversation/test_it_items.jsonl"


echo "stage2: instruction tuning"


NNODES=2
GPU_PER_NODES=8
PRIORITY=NORMAL # HIGHEST
EXP=it_full_deepspeed_multinode

IMG=XXXXX
PARTITION_ID=YYYY
# PARTITION_ID=vigen-2
WORKSPACE_ID=ZZZZ
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

CMD=" /root/pengyang/codebase/SOLAMI/models/vla/anygpt/train/conv_instruction_tuning.py \
    --deepspeed "/root/pengyang/codebase/SOLAMI/models/vla/scripts/stage2_deepspeed_zero3.json" \
    --run_name "it_full_deepspeed" \
    --model_name_or_path '${METAROOT}' \
    --it_data_path '${it_datasets}' \
    --cache_dir ${CACHEROOT} \
    --it_type 'full' \
    --use_lora False \
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
    --gradient_accumulation_steps 3 \
    --val_set_size 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \
    --use_flash_attn False \
    --overwrite_output_dir True\
    --save_total_limit 15" 

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
    --container-image ${IMG} \
     bash -c " conda activate /mnt/AFS_jiangjianping/miniconda3/envs/Final && export HOME='/mnt/AFS_jiangjianping' && cd "${HOME_DIR}"; python -m torch.distributed.run --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT --nproc-per-node "${GPU_PER_NODES}" --nnodes \$WORLD_SIZE --node_rank \$RANK ${CMD}"

