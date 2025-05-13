import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla')
# sys.path.append("/root/pengyang/codebase/SOLAMI/models/motiongpt")
import torch
import torch.nn as nn
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
import json
import re
import traceback
import torchaudio
from einops import rearrange
import yaml
from tqdm import tqdm

sys.path.append('tools/smplx')
import smplx

from torch.utils.data import Dataset as DatasetTorch
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from speechtokenizer import SpeechTokenizer
from transformers import  LlamaForCausalLM, LlamaTokenizer, GenerationConfig, EncodecModel, AutoProcessor
from m_utils.prompter import *
from m_utils.anything2token import *
from voice_clone import load_soundstorm, semantic2acoustic
from infer.pre_post_process import extract_text_between_tags
from motion.vqvae import VQVae, VQVAE_Trans
from motion.smplx_process import recover_from_smplx_feature
from motion.motion_dataset import MotionTextDataset
from collections import OrderedDict

DEBUG=False
LENGTHS=300


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

def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

class ECModel(nn.Module):
    def __init__(
        self, 
        model_name_or_path,
        output_dir,
        speech_tokenizer_path,
        speech_tokenizer_config,
        soundstorm_path,
        motion_tokenizer_dir,):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.prompter = Prompter()
        print("loading speech tokenzier")
        self.speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)     
        self.speech_tokenizer.eval()
        self.speech_tokenizer.to(device=self.device)
        self.soundstorm = load_soundstorm(soundstorm_path)
        self.soundstorm.eval()
        self.soundstorm.to(device=self.device)
        
        print("loading motion")
        body_config = open_yaml("/root/pengyang/codebase/SOLAMI/models/vla/motion/body.yaml")
        hand_config = open_yaml("/root/pengyang/codebase/SOLAMI/models/vla/motion/hand.yaml")
        self.vae_body = VQVae(**body_config)
        self.vae_hand = VQVae(**hand_config)
        trans_config = open_yaml("/root/pengyang/codebase/SOLAMI/models/vla/motion/trans.yaml")
        self.vae_transform = VQVAE_Trans(**trans_config)

        for key in ['vae_body', 'vae_hand', 'vae_transform']:
            file_name = key.split('_')[-1] + '.pth'
            file_path = os.path.join(motion_tokenizer_dir, file_name)
            state_dict = torch.load(file_path, map_location="cpu")
            getattr(self, key).load_state_dict(state_dict, strict=True)
            getattr(self, key).eval()
            getattr(self, key).to(device=self.device)
        
        self.load_motion_utils()
        
        # model
        print("loading llm")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )
        self.model.half()  
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

     
    def load_motion_utils(self):
        mean_var_path = "SOLAMI_data/mean_variance/all_mean_variance_post.npz"
        mean_var = np.load(mean_var_path, allow_pickle=True)
        motion_smplx = mean_var['smplx_feature'].item()
        self.motion_mean = np.concatenate([motion_smplx['root_velocity']['mean'], 
                                            motion_smplx['root_height']['mean'],
                                            motion_smplx['global_root_cont6d']['mean'],
                                            motion_smplx['cont6d_local']['mean'].reshape(-1)], axis=0)
        self.motion_std = np.concatenate([motion_smplx['root_velocity']['std'],
                                                    motion_smplx['root_height']['std'],
                                                    motion_smplx['global_root_cont6d']['std'],
                                                    motion_smplx['cont6d_local']['std'].reshape(-1)], axis=0)
        self.motion_std = np.where(self.motion_std == 0, 1e-9, self.motion_std)
        
        transforms = mean_var['transforms'].item()
        self.transform_mean = np.concatenate([transforms['smplx_relative_cont6d']['mean'], transforms['smplx_relative_pos']['mean']], axis=0)
        self.transform_std = np.concatenate([transforms['smplx_relative_cont6d']['std'], transforms['smplx_relative_pos']['std']], axis=0)
        self.transform_std_part = self.transform_std[[0, 2, 6, 7, 8]]
        self.transform_mean_part = self.transform_mean[[0, 2, 6, 7, 8]]
        self.betas = torch.tensor([-0.06134899, -0.4861751 ,  0.8630473 , -3.07320443,  1.10772016,
                                    -1.44656493,  2.97690664, -1.12731489,  1.24817344, -1.4111463 ,
                                    -0.04035034, -0.29547926,  0.38509519,  0.13750311,  0.94445029,
                                    -0.47172116], dtype=torch.float32)
        
        self.t_root_J = torch.tensor([
            0, 0, 0
        ], dtype=torch.float32)
        self.model_path = 'SOLAMI_data/SMPL_MODELS/smplx/SMPLX_MALE.npz'
        self.smplx_model = smplx.create(self.model_path, 
                                        model_type='smplx', 
                                        gender='male', 
                                        ext='npz', 
                                        num_betas=len(self.betas), 
                                        use_pca=False, 
                                        flat_hand_mean=True)
        self.smplx_model.eval()
        
        return
        
        
    def encode_speech(
        self,
        audio_path
    ):
        wav, sr = torchaudio.load(audio_path)
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1, ]
        if sr != self.speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).to(self.device)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = self.speech_tokenizer.encode(wav) # codes: (n_q, B, T)
        return codes[0, 0, :]
    
    def decode_speech(self, content, prompt_path=None):
        if prompt_path:
            # get tokens of prompt
            prompt_wav, sr = torchaudio.load(prompt_path)
            prompt_wav = prompt_wav.to(self.device)
            if sr != self.speech_tokenizer.sample_rate:
                prompt_wav = torchaudio.functional.resample(prompt_wav, sr, self.speech_tokenizer.sample_rate)
            # If it is stereo, take the average to mono
            if prompt_wav.shape[0] == 2:
                prompt_wav = prompt_wav.mean(dim=0).unsqueeze(0)
            prompt_tokens = rearrange(self.speech_tokenizer.encode(prompt_wav.unsqueeze(0)), 'q b n -> b n q')
        else:
            prompt_tokens = None
        # print(prompt_tokens)
        # codes.shape：(1, 1, n)
        semantic_codes = [[int(num) for num in re.findall(r'\d+', content)]]
        # wav: (b, 1, t)
        config_dict = json.load(open('/root/pengyang/codebase/SOLAMI/models/vla/config/generate_config.json', 'r'))
        wav = semantic2acoustic(torch.Tensor(semantic_codes).int().to(self.device), prompt_tokens, 
                                self.soundstorm, self.speech_tokenizer, steps=config_dict['vc_steps'])
        wav = wav.squeeze(0).detach().cpu()
        return wav
    
    
    def get_motion_code_from_str(self, motion_str, re_format):
        motion_ids = re.findall(re_format, motion_str)
        tmp_ids = [int(i) for i in motion_ids]
        output = torch.tensor(tmp_ids, dtype=int).to(self.device).reshape(1, -1)
        return output
    
    
    def smplx_infer(self, res):
        batch_size, seq_len, feat_len = res.shape
        
        global_orient = res[..., 3:6]
        
        transl = res[..., :3] - self.t_root_J.to(res.device)
        
        betas = self.betas.to(res.device)
        betas = betas.repeat(batch_size, seq_len, 1)
        
        expression = torch.zeros([batch_size, seq_len, 10], dtype=torch.float32).to(res.device)
        
        body_pose = res[..., 6:6+21*3]
        left_hand_pose = res[..., 6+21*3: 6+(21+15)*3]
        right_hand_pose = res[..., 6+36*3:]
        jaw_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
        leye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
        reye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
        
        body_parms = {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'jaw_pose': jaw_pose,
            'leye_pose': leye_pose,
            'reye_pose': reye_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'transl': transl,
            'betas': betas,
            'expression': expression,
        }
        
        for key in body_parms:
            body_parms[key] = body_parms[key].reshape(-1, body_parms[key].shape[-1])
        
        self.smplx_model.to(res.device)
        with torch.no_grad():
            output = self.smplx_model(**body_parms)
            
        joints = output.joints
        joints = joints.reshape(batch_size, seq_len, -1, 3)
        return joints[:, :, :55]
    
    
    def decode_motion(self, body_str, hand_str):
        body_codes = self.get_motion_code_from_str(body_str, r'<🕺(\d+)>')
        hand_codes = self.get_motion_code_from_str(hand_str, r'<🤚(\d+)>')
        min_len = min(body_codes.shape[1], hand_codes.shape[1])
        body_codes = body_codes[:, :min_len]
        hand_codes = hand_codes[:, :min_len]
        if min_len == 0:
            return None, None
        else:
            body_repre = self.vae_body.decode(body_codes)
            hand_repre = self.vae_hand.decode(hand_codes)
            decode_repre = torch.cat((body_repre, hand_repre), dim=-1)
            mean = torch.tensor(self.motion_mean).to(decode_repre)
            std = torch.tensor(self.motion_std).to(decode_repre)
            decode_repre = decode_repre * std + mean
            smplx_params = recover_from_smplx_feature(decode_repre, 'local cont6d')
            smplx_joints = self.smplx_infer(smplx_params)
            return smplx_joints.squeeze(0).detach().cpu(), smplx_params.squeeze(0).detach().cpu()
    
    
    def preprocess(
        self,
        input_data,
        modality,
        to_modality,
    ):
        # processed_parts = []
        if modality == "text":
            processed_inputs = input_data
        else:
            if modality == "image":
                tokens = self.encode_image(image_path=input_data.strip())[0]
            elif modality == "speech":
                tokens = self.encode_speech(input_data.strip()) # speechtokenizer
            else:
                raise TypeError("wrong modality")
            processed_inputs = modality_tokens_to_string(tokens=tokens, modality=modality)
        prompt_seq = self.prompter.generate_prompt_input(modality_str=processed_inputs, modality=modality,
                                                         to_modality=to_modality)
        return prompt_seq

    
    def modality_str_to_input_token_ids(
        self,
        task,
        motion1=None, 
        motion2=None,
        text=None,
        speech1=None,
        speech2=None,
    ):
        if task == 't2m':
            prompt_str = self.prompter.generate_prompt_input(
                modality='text',
                modality_str=text,
                to_modality='motion',
            )
        elif task == 'm2t':
            ## smplify the code
            motion_str = ''
            for item in motion1:
                motion_str += item
            
            prompt_str = self.prompter.generate_prompt_input(
                modality='motion',
                modality_str=motion_str,
                to_modality='text',
            )
        elif task == 'm2m':
            motion_str = ''
            for item in motion1:
                motion_str += item
            prompt_str = self.prompter.generate_prompt_input(
                modality='motion',
                modality_str=motion_str,
                to_modality='motion',
            )
        elif task == 't2s':
            prompt_str = self.prompter.generate_prompt_input(
                modality='text',
                modality_str=text,
                to_modality='speech',
            )
        elif task == 's2t':
            prompt_str = self.prompter.generate_prompt_input(
                modality='speech',
                modality_str=speech1,
                to_modality='text',
            )
        elif task == 's2s':
            prompt_str = self.prompter.generate_prompt_input(
                modality='speech',
                modality_str=speech1,
                to_modality='speech',
            )
        else:
            raise ValueError("wrong task")
        
        input_ids = self.tokenizer(prompt_str, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)
        return input_ids
    
    
    def llm_inference(
        self,
        input_ids,
        task,
    ):
        config_path='/root/pengyang/codebase/SOLAMI/models/vla/config/text_generate_config.json'
        if task in ['t2m', 'm2m']:
            config_path='/root/pengyang/codebase/SOLAMI/models/vla/config/motion_generate_config.json'
        elif task in ['t2s', 's2s']:
            config_path='/root/pengyang/codebase/SOLAMI/models/vla/config/speech_generate_config.json'
        config_dict = json.load(open(config_path, 'r'))
        generation_config = GenerationConfig(    
            **config_dict
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = generated_ids.sequences
        response = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
        
        return response
    
    
    def extract_LLM_part(self, text: str, N: int=-1):
        if N == -1:
            match = re.findall(r'[Human](.*)', text)
            if len(match) > 0:
                return match[-1].strip()
            else:
                return ''
        else:
            match = re.findall(r'[MMGPT](.*?)[Human]', text)
            if N < len(match):
                return match[N].strip()
            else:
                return ''
    
    
    def llm_response_extract(
        self,
        response,
        task,
    ):
        output_str = self.extract_LLM_part(response)
        if task in ['t2m', 'm2m']:
            contents = []
            for modality in ['body', 'hand']: 
                special_dict = modal_special_str[modality]
                modality_content = extract_text_between_tags(response, tag1=special_dict['sos'], tag2=special_dict['eos'])
                contents.append(modality_content)
            return contents
        elif task in ['t2s', 's2s']:
            special_dict = modal_special_str['speech']
            modality_content = extract_text_between_tags(response, tag1=special_dict['sos'], tag2=special_dict['eos'])
            return [modality_content, ]
        else:
            content = extract_text_between_tags(response, tag1=f"{chatbot_name} : ", tag2="<eos>").strip()
        return [content,]
        pass
    
    
    
    def modality_decode(
        self, 
        content,
        task,
        prompt_path=None,
    ):
        if task in ['t2m', 'm2m']:
            hand_content = content[1]
            body_content = content[0]
            smplx_joints, smplx_params = self.decode_motion(body_content, hand_content)
            return smplx_joints, smplx_params
        elif task in ['t2s', 's2s']:
            speech_content = content[0]
            generated_wav = self.decode_speech(speech_content, prompt_path)
            # torchaudio.save("/root/pengyang/codebase/SOLAMI/models/vla/infer_output/pretrain_checkpoint-4096/test.wav", generated_wav, self.speech_tokenizer.sample_rate)
            return generated_wav
        else:
            text_content = content[0]
            return text
        pass
    
    
    def pretrain_inference_with_token_ids(
        self,
        task,
        motion1=None,
        motion2=None,
        speech1=None,
        speech2=None,
        text=None,):
        if task == 'm2m' and motion2 is None:
            return None
        if task == 's2s' and speech2 is None:
            return None
        
        if task in ['t2m', 'm2t', 'm2m',]:
            if motion1 is None:
                return None
            input_ids = pretrained_model.modality_str_to_input_token_ids(
                task=task,
                motion1=motion1, 
                motion2=motion2,
                text=text,
            )
        elif task in ['t2s', 's2t', 's2s']:
            if speech1 is None:
                return None
            input_ids = pretrained_model.modality_str_to_input_token_ids(
                task=task,
                text=text,
                speech1=speech1,
                speech2=speech2,
            )
        else:
            print("wrong task")
            pass
        response = pretrained_model.llm_inference(
            input_ids=input_ids,
            task=task,
        )
        # print(response)
        
        modality_extract = pretrained_model.llm_response_extract(
            response=response,
            task=task,
        )
        # print(modality_extract)
        
        output = pretrained_model.modality_decode(
            content=modality_extract,
            task=task,
        )
        
        return output
        
    
    
    def forward(
        self, 
        prompts
    ):
       pass
        


class MotionDataset(DatasetTorch):
    def __init__(self, 
                 data_path: str,
                prompter,
                raw_dataset=None,):
        super(MotionDataset, self).__init__()
        if raw_dataset is not None:
            self.raw_dataset = raw_dataset
        else:
            raw_dataset = load_dataset("json", data_files=data_path)
            raw_dataset = raw_dataset['train']
            if DEBUG:
                raw_dataset = raw_dataset.select(range(LENGTHS))
            self.raw_dataset = raw_dataset
        print("Loaded {} items from dataset {}".format(len(self.raw_dataset), data_path))
        # self.tokenizer = tokenizer
        self.prompter = prompter

    def __len__(self):
        return len(self.raw_dataset)


    def __getitem__(self, i):
        raw_data = self.raw_dataset[i]
        chat_data = raw_data['chat']
        
        body1 = chat_data[0]['body']
        hand1 = chat_data[0]['hand']
        trans1 = chat_data[0]['trans']
        motion1 = [
            modality_tokens_to_string(trans1, modality="trans"),
            modality_tokens_to_string(body1, modality="body"),
            modality_tokens_to_string(hand1, modality="hand"),
        ]
        text = random.choice(chat_data[0]['text'])
        if len(chat_data) > 1:
            body2 = chat_data[1]['body']
            hand2 = chat_data[1]['hand']
            motion2 = [
                modality_tokens_to_string(body2, modality="body"),
                modality_tokens_to_string(hand2, modality="hand"),
            ]
            text2 = random.choice(chat_data[1]['text'])
        else:
            motion2 = None
            text2=None
        
        speech1 = None
        speech2 = None
        return text, motion1, motion2, speech1, speech2, text2, raw_data



class SpeechDataset(DatasetTorch):
    def __init__(self, 
                 data_path: str,
                prompter,
                raw_dataset=None,):
        super(SpeechDataset, self).__init__()
        if raw_dataset is not None:
            self.raw_dataset = raw_dataset
        else:
            print("Loading from dataset {}".format(data_path))
            raw_dataset = load_dataset("json", data_files=data_path)
            raw_dataset = raw_dataset['train']
            test_size = 1024
            split = len(raw_dataset) - test_size
            raw_dataset = raw_dataset.select(range(split, len(raw_dataset)))
            # TODO debug
            if DEBUG:
                raw_dataset = raw_dataset.select(range(LENGTHS))
            self.raw_dataset = raw_dataset
        print("Loading {} items from dataset {}".format(len(self.raw_dataset), data_path))
        self.prompter = prompter

    def __len__(self):
        return len(self.raw_dataset)


    def __getitem__(self, i):
        raw_data = self.raw_dataset[i]
        chat_data = raw_data['chat']
        
        text = chat_data[0]['text']
        speech1 = chat_data[0]['speech']
        if len(chat_data) > 1:
            speech2 = chat_data[1]['speech']
            text2 = chat_data[1]['text']
        else:
            speech2 = None
            text2 = None
        motion1 = None
        motion2 = None
        return text, motion1, motion2, speech1, speech2, text2, raw_data


    
if __name__ == "__main__":
    ## argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--period", type=int, default=4)
    args = parser.parse_args()
    
    model_name_or_path = "/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain_audio_motion/checkpoint-4096"
    output_dir = "/root/pengyang/codebase/SOLAMI/models/vla/infer_output/pretrain_checkpoint-4096-final"
    speech_tokenizer_path = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/ckpt.dev"
    speech_tokenizer_config = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/config.json"
    soundstorm_path = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/soundstorm/speechtokenizer_soundstorm_mls.pt"
    motion_tokenizer_dir = "/root/pengyang/codebase/SOLAMI/extra/motion_tokenizer"
    
    prompter = Prompter()
    
    pretrained_model = ECModel(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        speech_tokenizer_path=speech_tokenizer_path,
        speech_tokenizer_config=speech_tokenizer_config,
        soundstorm_path=soundstorm_path,
        motion_tokenizer_dir=motion_tokenizer_dir,
    )
    
    # raw_motion_dataset = MotionTextDataset(
    #     mean=pretrained_model.motion_mean,
    #     std=pretrained_model.motion_std,
    #     transform_mean=pretrained_model.transform_mean,
    #     transform_std=pretrained_model.transform_std,
    #     tmpFile=True,
    #     tiny=False,
    # )
    
    motion_test_data_path = "SOLAMI_data/tmp_data/pretrain_tokens/local_cont6d_body_hand_sep/motion_test_merged.jsonl"
    motion_eval_dataset = MotionDataset(
        data_path=motion_test_data_path,
        prompter=prompter,
    )
    print(len(motion_eval_dataset))
    
    speech_test_data_path = "SOLAMI_data/audio/anyinstruct/anyinstruct_0_4.jsonl"
    speech_eval_dataset = SpeechDataset(
        data_path=speech_test_data_path,
        prompter=prompter,
    )
    
    # task = 's2s'
    # for data_item in tqdm(speech_eval_dataset):
    #     text1, motion1, motion2, speech1, speech2, text2, chat_data = data_item
    #     print(text, motion1, motion2, speech1, speech2)
    #     results = pretrained_model.pretrain_inference_with_token_ids(
    #         task=task,
    #         speech1=speech1, 
    #         speech2=speech2,
    #         text=text1,
    #     )
    #     pass

    dataset_dicts = {
        'motion': motion_eval_dataset,
        # 'speech_anyinstruct': speech_eval_dataset,
    }
    count = 0
    for dataset_name, eval_dataset in dataset_dicts.items():
        count += 1
        dataset_save_dir = os.path.join(output_dir, dataset_name)
        if not os.path.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir, exist_ok=True)
        for idx in tqdm(range(args.part, len(eval_dataset), args.period)):
            data_item = eval_dataset[idx]
            text, motion1, motion2, speech1, speech2, text2, chat_data = data_item
            
            #### check VQVAE 2999 is right
            # print(text, motion1, motion2, speech1, speech2)
            # body_codes = pretrained_model.get_motion_code_from_str(motion1[1], r'<🕺(\d+)>')
            # hand_codes = pretrained_model.get_motion_code_from_str(motion1[2], r'<🤚(\d+)>')          
            # body_repre = pretrained_model.vae_body.decode(body_codes)
            # hand_repre = pretrained_model.vae_hand.decode(hand_codes)
            # decode_repre = torch.cat((body_repre, hand_repre), dim=-1)
            # mean = torch.tensor(pretrained_model.motion_mean).to(decode_repre)
            # std = torch.tensor(pretrained_model.motion_std).to(decode_repre)
            # decode_repre = decode_repre * std + mean
            # smplx_params = recover_from_smplx_feature(decode_repre, 'local cont6d')
            # smplx_joints = pretrained_model.smplx_infer(smplx_params)
            # ### save smplx_joints
            # smplx_joints = smplx_joints.squeeze(0).detach().cpu()
            # np.save(os.path.join('/root/pengyang/codebase/SOLAMI/models/vla/infer_output/test_vqvae_2999', chat_data['id'] + '_motion.npy'), smplx_joints)
            # print(chat_data['id'])
            # print(text)
            # pass
            save_res = {}
            for task in ['t2m', 'm2t', 'm2m', 't2s', 's2t', 's2s']:
                results = pretrained_model.pretrain_inference_with_token_ids(
                    task=task,
                    motion1=motion1, 
                    motion2=motion2,
                    text=text,
                    speech1=speech1,
                    speech2=speech2,
                )
                if results is None:
                    continue
                save_res[task] = {}
                if task == 't2m':
                    save_res[task]['gt'] = chat_data['chat'][0]['motion_id']
                    save_res[task]['pred'] = results[-1].numpy() if results[-1] is not None else None
                if task == 'm2m':
                    save_res[task]['gt'] = chat_data['chat'][1]['motion_id']
                    save_res[task]['pred'] = results[-1].numpy() if results[-1] is not None else None
                if task == 'm2t':
                    save_res[task]['gt'] = text
                    save_res[task]['pred'] = results
                if task == 't2s':
                    save_res[task]['gt'] = chat_data['chat'][0]['speech_path']
                    save_res[task]['pred'] = results[-1].numpy() if results[-1] is not None else None
                if task == 's2t':
                    save_res[task]['gt'] = text
                    save_res[task]['pred'] = results
                if task == 's2s':
                    save_res[task]['gt'] = chat_data['chat'][1]['speech_path']
                    save_res[task]['pred'] = results[-1].numpy() if results[-1] is not None else None
                # save as npz
                
            save_path = os.path.join(dataset_save_dir, chat_data['id'] + '.npz')
            np.savez(save_path, **save_res)
            