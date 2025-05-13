import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla')
os.environ["TRANSFORMERS_CACHE"] = "~/.cache/huggingface/hub"
import numpy as np
import torch
sys.path.append('tools/smplx')
import smplx
from motion.motion_dataset import MotionTextDataset
from metrics.motion_metrics import MTMetrics
from metrics.speech_metrics import STMetrics
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import whisper
import torchaudio

import torch.distributed as dist
import debugpy

def initialize_debugpy():
    if not dist.is_initialized() or dist.get_rank() == 0:
        # print(f"Rank: {dist.get_rank()} - Debugpy is listening on port 15696")
        print("Debugpy is listening on port 15696")
        debugpy.listen(("0.0.0.0", 15696))
        debugpy.wait_for_client()
        
# def initialize_distributed():
#     if not dist.is_initialized():
#         dist.init_process_group(backend='nccl')

# initialize_distributed()
# initialize_debugpy()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_motion_utils():
    mean_var_path = "SOLAMI_data/mean_variance/all_mean_variance_post.npz"
    mean_var = np.load(mean_var_path, allow_pickle=True)
    motion_smplx = mean_var['smplx_feature'].item()
    motion_mean = np.concatenate([motion_smplx['root_velocity']['mean'], 
                                        motion_smplx['root_height']['mean'],
                                        motion_smplx['global_root_cont6d']['mean'],
                                        motion_smplx['cont6d_local']['mean'].reshape(-1)], axis=0)
    motion_std = np.concatenate([motion_smplx['root_velocity']['std'],
                                                motion_smplx['root_height']['std'],
                                                motion_smplx['global_root_cont6d']['std'],
                                                motion_smplx['cont6d_local']['std'].reshape(-1)], axis=0)
    motion_std = np.where(motion_std == 0, 1e-9, motion_std)
    
    transforms = mean_var['transforms'].item()
    transform_mean = np.concatenate([transforms['smplx_relative_cont6d']['mean'], transforms['smplx_relative_pos']['mean']], axis=0)
    transform_std = np.concatenate([transforms['smplx_relative_cont6d']['std'], transforms['smplx_relative_pos']['std']], axis=0)
    transform_std_part = transform_std[[0, 2, 6, 7, 8]]
    transform_mean_part = transform_mean[[0, 2, 6, 7, 8]]
    betas = torch.tensor([-0.06134899, -0.4861751 ,  0.8630473 , -3.07320443,  1.10772016,
                                -1.44656493,  2.97690664, -1.12731489,  1.24817344, -1.4111463 ,
                                -0.04035034, -0.29547926,  0.38509519,  0.13750311,  0.94445029,
                                -0.47172116], dtype=torch.float32)
    
    t_root_J = torch.tensor([
        0, 0, 0
    ], dtype=torch.float32)
    return motion_mean, motion_std, transform_mean_part, transform_std_part, betas, t_root_J




data_dirs = [
    # "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/pretrain_checkpoint-4096/motion",
    # "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/pretrain_checkpoint-4096/speech_anyinstruct",
    "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/pretrain_checkpoint-4096-final/motion",
    ]


motion_mean, motion_std, transform_mean, transform_std, betas, t_root_J = get_motion_utils()

raw_motion_dataset = MotionTextDataset(
    mean=motion_mean,
    std=motion_std,
    transform_mean=transform_mean,
    transform_std=transform_std,
    tmpFile=True,
    tiny=False,
)

motion_metrics = {
    't2m' : MTMetrics(diversity_times=50, device=DEVICE, task='t2m'),
    'm2m' : MTMetrics(diversity_times=50, device=DEVICE, task='m2m'),
    # 'm2t' : MTMetrics(diversity_times=50, device=DEVICE, task='m2t'),
    }

vc_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')#.to(DEVICE)
vc_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(DEVICE)
whisper_model = whisper.load_model('large-v3', download_root='~/.cache/whisper', device=DEVICE)

speech_metrics = {
    't2s' : STMetrics(device=DEVICE, task='t2s', vc_feature_extractor=vc_feature_extractor, vc_model=vc_model, whisper_model=whisper_model),
    's2s' : STMetrics(device=DEVICE, task='s2s', vc_feature_extractor=vc_feature_extractor, vc_model=vc_model, whisper_model=whisper_model),
    's2t' : STMetrics(device=DEVICE, task='s2t', vc_feature_extractor=vc_feature_extractor, vc_model=vc_model, whisper_model=whisper_model),
}


for data_dir in data_dirs:
    dataset_name = os.path.basename(data_dir)
    print(f"Processing {dataset_name}...")
    ### load npz data, data is dict
    
    data_file_names = os.listdir(data_dir)
    for data_file_name in data_file_names:
        data_file_path = os.path.join(data_dir, data_file_name)
        data = np.load(data_file_path, allow_pickle=True)
        data = dict(data)
        for key in data.keys():
            if key in ['t2m', 'm2m']:
                gt_motion_name = data[key].item()['gt']
                gt_smplx_params = raw_motion_dataset.get_data_from_name(gt_motion_name)[1]
                pred_smplx_params = data[key].item()['pred']
                motion_metrics[key].update(feats_ref=gt_smplx_params, feats_rst=pred_smplx_params, task=key)
            elif key == 'm2t':
                # gt_motion_name = data[key].item()['gt']
                # pred_smplx_params = data[key].item()['pred']
                # motion_metrics[key].update(gt_smplx_params, pred_smplx_params)
                # motion_metrics[key].update(gt_texts=data[key].item()['gt'], pred_texts=data[key].item()['pred'], task=key)
                pass
            elif key in ['t2s', 's2s']:
                gt_speech_name = data[key].item()['gt']
                gt_wav, gt_sr = torchaudio.load(gt_speech_name)
                if gt_sr != 16000:
                    gt_wav = torchaudio.functional.resample(gt_wav, gt_sr, 16000)
                gt_wav = gt_wav[0]
                if data[key].item()['pred'] is not None:
                    pred_wav = torch.tensor(data[key].item()['pred'], dtype=torch.float32)
                    # pred_wav = torchaudio.functional.resample(pred_wav, 24000, 16000)
                else:
                    pred_wav = None
                speech_metrics[key].update(target=gt_wav, pred=pred_wav, task=key)
                pass
            elif key in ['s2t']:
                speech_metrics[key].update(target=data[key].item()['gt'], pred=data[key].item()['pred'], task=key)
                pass
    for key in motion_metrics.keys():
        metrics = motion_metrics[key].compute()
        # log metrics
        print('='*10)
        print("For dataset {} task {}".format(dataset_name, key))
        motion_metrics[key].log_metrics(metrics)
    
    for key in speech_metrics.keys():
        metrics = speech_metrics[key].compute()
        # log metrics
        print('='*10)
        print("For dataset {} task {}".format(dataset_name, key))
        speech_metrics[key].log_metrics(metrics)
    break
    