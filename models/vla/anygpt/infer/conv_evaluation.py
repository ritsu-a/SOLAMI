import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla')
os.environ["TRANSFORMERS_CACHE"] = "~/.cache/huggingface/hub"
import numpy as np
import torch
sys.path.append('tools/smplx')
import smplx
from tqdm import tqdm
from motion.motion_dataset import MotionTextDataset
from metrics.motion_metrics import MTMetrics
from metrics.speech_metrics import STMetrics
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import whisper
import torchaudio
import json
import argparse
import torch.distributed as dist
import debugpy


BETAS = np.array([-0.06134899, -0.4861751 ,  0.8630473 , -3.07320443,  1.10772016,
                                    -1.44656493,  2.97690664, -1.12731489,  1.24817344, -1.4111463 ,
                                    -0.04035034, -0.29547926,  0.38509519,  0.13750311,  0.94445029,
                                    -0.47172116])

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
    # model_path = '/mnt/AFS_jiangjianping/datasets/Assets/SMPL_MODELS/smplx/SMPLX_MALE.npz'
    # smplx_model = smplx.create(model_path, 
    #                                 model_type='smplx', 
    #                                 gender='male', 
    #                                 ext='npz', 
    #                                 num_betas=len(betas), 
    #                                 use_pca=False, 
    #                                 flat_hand_mean=True)
    # smplx_model.eval()
    return motion_mean, motion_std, transform_mean_part, transform_std_part, betas, t_root_J


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/pengyang/codebase/SOLAMI/models/vla/infer_output/it_full_checkpoint-128")
    parser.add_argument("--save_gt", type=bool, default=False)
    parser.add_argument("--save_pred", type=bool, default=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    speech_dir = "SOLAMI_data/IT_Speech/raw_data"

    motion_mean, motion_std, transform_mean, transform_std, betas, t_root_J = get_motion_utils()

    raw_motion_dataset = MotionTextDataset(
        mean=motion_mean,
        std=motion_std,
        transform_mean=transform_mean,
        transform_std=transform_std,
        tmpFile=True,
        tiny=False,
    )

    # raw_motion_dataset = None

    vc_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')#.to(DEVICE)
    vc_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(DEVICE)
    whisper_model = whisper.load_model('large-v3', download_root='~/.cache/whisper', device=DEVICE)

    motion_metric = MTMetrics(diversity_times=50, device=DEVICE, task='t2m')
    speech_metric = STMetrics(device=DEVICE, task='t2s', vc_feature_extractor=vc_feature_extractor, vc_model=vc_model, whisper_model=whisper_model)

    gt_data_file = "SOLAMI_data/Conversation/test_it_items.jsonl"
    ### read jsonl file
    test_data_dict = {}
    with open(gt_data_file, 'r') as f:
        for line in f:
            data_item = json.loads(line)
            test_data_dict[data_item['id']] = data_item
    print(f"Total test data items: {len(test_data_dict)}")

    output_dir = data_dir + '_evaluation'
    os.makedirs(output_dir, exist_ok=True)

    ### get all the files
    data_file_names = os.listdir(data_dir)
    for data_file_name in tqdm(data_file_names):
        if not data_file_name.endswith('.npz'):
            continue
        data_file_path = os.path.join(data_dir, data_file_name)
        
        conv_id = data_file_name.split('.')[0]
        
        current_data_dir = os.path.join(output_dir, conv_id)
        os.makedirs(current_data_dir, exist_ok=True)
     
        gt_data_item = test_data_dict[conv_id]

                
        dialogs_record = {}
        ### record id, topic, role, chat: 
        dialogs_record['id'] = gt_data_item['id']
        dialogs_record['topic'] = gt_data_item['topic']
        dialogs_record['role'] = gt_data_item['chat'][1]['role']
        dialogs_record['chat'] = {}
        for r_id, chat_item in enumerate(gt_data_item['chat']):
            dialogs_record['chat'][str(r_id)] = {}
            dialogs_record['chat'][str(r_id)]['gt'] = {
                "speech_text": chat_item['speech_text'],
                "speech_file_name": chat_item['speech_file_name'],
                'role': chat_item['role'],
                'motion_des': chat_item['motion_des'],
                'motion_id': chat_item['motion_id'],
            }
            if args.save_gt:
                gt_speech_path = os.path.join(speech_dir, chat_item['speech_file_name'])
                gt_wav, gt_sr = torchaudio.load(gt_speech_path)
                if gt_sr != 16000:
                    gt_wav = torchaudio.functional.resample(gt_wav, gt_sr, 16000)
                torchaudio.save(os.path.join(current_data_dir, 'gt_' + str(r_id)+'.wav'), gt_wav, 16000)
                gt_motion_id = chat_item['motion_id']
                gt_smplx_params = raw_motion_dataset.get_data_from_name(gt_motion_id)[1]
                if gt_smplx_params is None:
                    continue
                trans = gt_smplx_params[:, :3]
                poses = gt_smplx_params[:, 3:]
                smplx_poses = np.concatenate([poses, np.zeros((poses.shape[0], 9))], axis=-1)
                gt_motion_params_np = {
                    'betas': BETAS,
                    'gender': 'male',
                    'mocap_framerate': 30,
                    'poses': smplx_poses,
                    'trans': trans,
                }
                np.savez(os.path.join(current_data_dir, 'gt_' + str(r_id)+'.npz'), **gt_motion_params_np)
            pass
        
        data = np.load(data_file_path, allow_pickle=True)
        data = dict(data)
        for key in ['0', '2', '4', '6', '8']:
            if key in data:
                round_id = int(key) + 1
                ### speech
                gt_speech_name = gt_data_item['chat'][round_id]['speech_file_name']
                gt_speech_path = os.path.join(speech_dir, gt_speech_name)
                gt_wav, gt_sr = torchaudio.load(gt_speech_path)
                if gt_sr != 16000:
                    gt_wav = torchaudio.functional.resample(gt_wav, gt_sr, 16000)
                gt_wav = gt_wav[0]
                
                pred_wav = torch.tensor(data[key].item()['speech'], dtype=torch.float32)[0]
                # pred_wav = torchaudio.functional.resample(pred_wav, 24000, 16000)
                # output_audio_dir = data_dir + '_output_audios'
                # if not os.path.exists(output_audio_dir):
                #     os.makedirs(output_audio_dir)
                # torchaudio.save(os.path.join(output_audio_dir, conv_id + str(round_id)+'.wav'), pred_wav.unsqueeze(0), 16000)
                
                speech_metric.update(target=gt_wav, pred=pred_wav, task='t2s')
                
                ### motion
                gt_motion_id = gt_data_item['chat'][round_id]['motion_id']
                gt_smplx_params = raw_motion_dataset.get_data_from_name(gt_motion_id)[1]
                if gt_smplx_params is None:
                    continue
                pred_smplx_params = data[key].item()['motion_params']
                motion_metric.update(feats_ref=gt_smplx_params, feats_rst=pred_smplx_params, task='t2m')
                
                if data[key].item()['speech'] is not None:
                    dialogs_record['chat'][str(round_id)]['pred'] = {
                        'speech_text': speech_metric.pred_texts[-1],
                    }
                
                if args.save_pred:
                    torchaudio.save(os.path.join(current_data_dir, 'pred_' + str(round_id)+'.wav'), pred_wav.unsqueeze(0), 16000)
                    trans = pred_smplx_params[:, :3]
                    poses = pred_smplx_params[:, 3:]
                    pred_smplx_poses = np.concatenate([poses, np.zeros((poses.shape[0], 9))], axis=-1)
                    pred_motion_params_np = {
                        'betas': BETAS,
                        'gender': 'male',
                        'mocap_framerate': 30,
                        'poses': pred_smplx_poses,
                        'trans': trans,
                    }
                    np.savez(os.path.join(current_data_dir, 'pred_' + str(round_id)+'.npz'), **pred_motion_params_np)
                
                pass
            else:
                speech_metric.update(target=None, pred=None, task='t2s')
                motion_metric.update(feats_ref=None, feats_rst=None, task='t2m')
                pass
            
        ### save the record
        save_path = os.path.join(current_data_dir, 'record.json')
        with open(save_path, 'w') as f:
            json.dump(dialogs_record, f, indent=4)
        torch.cuda.empty_cache()
    m_metrics = motion_metric.compute()
    ret_m_metrics = motion_metric.log_metrics(m_metrics)
    s_metrics = speech_metric.compute()
    ret_s_metrics = speech_metric.log_metrics(s_metrics)
    ### save the metrics
    save_path = os.path.join(output_dir, 'metrics.json')
    with open(save_path, 'w') as f:
        json.dump({'motion': ret_m_metrics, 'speech': ret_s_metrics}, f, indent=4)
