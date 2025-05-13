#!/bin/bash



DATA_DIR=(
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-0_evaluation"
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-1_evaluation"
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-2_evaluation"
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-3_evaluation"
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2-dlp-motiongpt-retrieval-final-4_evaluation"

)
for i in "${!DATA_DIR[@]}"; do
    python speech_evaluation_gpt-4o.py --data_dir ${DATA_DIR[$i]} &
    wait
done
