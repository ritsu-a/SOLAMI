#!/bin/bash

declare -a paths=( \
            "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1/sim1__1200_0_c_sc.json" \
            "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1/sim1__1440_0_c_sc.json" \
            "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1/sim1__1520_0_c_sc.json" \
            )


for path in "${paths[@]}"
do
    echo "Processing path: $path"
    
    python tts.py --json_path "$path"
    
    /mnt/AFS_jiangjianping/tools/blender-3.6.10-linux-x64/blender -b -P blender_render.py -- --json "$path"
done