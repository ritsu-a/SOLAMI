import os
import argparse
import json
from pydub import AudioSegment
import numpy as np

def get_wav_length(file_path):
    audio = AudioSegment.from_wav(file_path)
    return audio.duration_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1/sim1__1200_0_if_sc.json')
    args = parser.parse_args()
    json_path = args.json_path
    
    with open(json_path, 'r') as f:
        conv_data = json.load(f)
    audio_dir = "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1_audio"
    basename = os.path.basename(json_path)
    audio_dir = os.path.join(audio_dir, basename.split('.')[0])
    
    m_dataset_items = {}
    item_paths = [
        "SOLAMI_data/HumanML3D/dataset_items.json",
        "SOLAMI_data/DLP-MoCap/dataset_items.json",
        "SOLAMI_data/Inter-X/dataset_items.json",
    ]
    for item_path in item_paths:
        with open(item_path, 'r') as f:
            m_dataset_items.update(json.load(f))
            
    motion_dirs = {
        'dlp': 'SOLAMI_data/DLP-MoCap',
        'humanml3d': 'SOLAMI_data/HumanML3D',
        'interx': 'SOLAMI_data/Inter-X',
    }
    
    align_dir = "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1_align"
    os.makedirs(align_dir, exist_ok=True)
    align_path = os.path.join(align_dir, basename)
    
    aligns = {}
    
    p_time = 0
    for round, dialog in conv_data[0]['dialogs'].items():
        audio_length = get_wav_length(os.path.join(audio_dir, f'{round}.wav'))
        motion_dataset_name = dialog['action_dataset']
        motion_path_tmp = m_dataset_items[dialog['action_index']]['motion_joints_path']
        motion_path = os.path.join(motion_dirs[motion_dataset_name], motion_path_tmp)
        motion = np.load(motion_path)
        motion_length = motion.shape[0] / 30
        max_duration = max(audio_length, motion_length)
        p_time += max_duration + 1
        aligns[round] = p_time
    
    with open(align_path, 'w') as f:
        json.dump(aligns, f, indent=2)
        