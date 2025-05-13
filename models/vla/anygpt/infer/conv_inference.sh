

MODEL_NAME_OR_PATH="/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain_audio_motion_final/checkpoint-2560"
LORA_MODEL_NAME_OR_PATH=(
    "/root/pengyang/codebase/SOLAMI/models/vla/output_models/it_lora_deepspeed/checkpoint-768"
)
OUTPUT_DIR=(
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_lora_checkpoint-768-final-1"
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_lora_checkpoint-768-final-2"
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_lora_checkpoint-768-final-3"
)
for i in "${!LORA_MODEL_NAME_OR_PATH[@]}"; do
    CUDA_VISIBLE_DEVICES=6,7 python conv_inference.py --part 0 --period 1 --model_name_or_path ${MODEL_NAME_OR_PATH} --lora_model_name_or_path ${LORA_MODEL_NAME_OR_PATH[$i]} --output_dir ${OUTPUT_DIR[$i]} &
    wait
done
# CUDA_VISIBLE_DEVICES=0,1 python conv_inference.py --part 0 --period 4 --model_name_or_path ${MODEL_NAME_OR_PATH} --lora_model_name_or_path ${LORA_MODEL_NAME_OR_PATH} --output_dir ${OUTPUT_DIR}\


CUDA_VISIBLE_DEVICES=6,7 python conv_inference.py --part 0 --period 1 --model_name_or_path "/root/pengyang/codebase/SOLAMI/models/vla/output_models/it_full_deepspeed_multinode_no_pretrain/checkpoint-768" --output_dir "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint_no_pretrain-768-final-latency-test" --use_vllm True &
wait

CUDA_VISIBLE_DEVICES=6,7 python conv_inference.py --part 0 --period 1 --model_name_or_path "/root/pengyang/codebase/SOLAMI/models/vla/output_models/it_full_deepspeed_multinode/checkpoint-768" --output_dir "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint-768-final-latency-latency-test" --use_vllm True &
wait

CUDA_VISIBLE_DEVICES=6,7 python conv_inference.py --part 0 --period 1 --model_name_or_path "/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain_audio_motion_final/checkpoint-2560" --lora_model_name_or_path "/root/pengyang/codebase/SOLAMI/models/vla/output_models/it_lora_deepspeed/checkpoint-768" --output_dir "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_lora_checkpoint-768-final-latency-test" &
wait 

echo "All processes have finished."