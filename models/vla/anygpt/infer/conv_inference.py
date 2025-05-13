import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla')
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import logging
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import Trainer
import copy
import argparse
import json
import re
import traceback
import torchaudio
from einops import rearrange
import yaml
from tqdm import tqdm
sys.path.append('tools/smplx')
import smplx
import time


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


from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from m_utils.loggings import get_logger
from m_utils.prompter import *
from m_utils.anything2token import *
from m_utils.conv import Conversation, SeparatorStyle


IGNORE_TOKEN_ID=-100
LLM_MAX_LENGTH = 2048
LORA=False
DEBUG=True
LENGTHS=30
REPEAT_TIMES=3

CHECK_GPU_USAGE=False

import torch.distributed as dist
import debugpy

def initialize_debugpy():
    # if not dist.is_initialized() or dist.get_rank() == 0:
        # print(f"Rank: {dist.get_rank()} - Debugpy is listening on port 15696")
        print("Debugpy is listening on port 15696")
        debugpy.listen(("0.0.0.0", 15697))
        debugpy.wait_for_client()
        
def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

# initialize_distributed()
# initialize_debugpy()

def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


class ITDataset(DatasetTorch):
    def __init__(self, 
                data_path: str,
                raw_dataset=None,):
        super(ITDataset, self).__init__()
        if raw_dataset is not None:
            self.raw_dataset = raw_dataset
        else:
            print("Loading from dataset {}".format(data_path.split('/')[-1]))
            raw_dataset = load_dataset("json", data_files=data_path)
            raw_dataset = raw_dataset['train']
            # TODO debug
            if DEBUG:
                raw_dataset = raw_dataset.select(range(LENGTHS))
            self.raw_dataset = raw_dataset
        print("Loading {} items from dataset {}".format(len(self.raw_dataset), data_path.split('/')[-1]))


    def __len__(self):
        return len(self.raw_dataset)


    def __getitem__(self, idx):
        raw_data = self.raw_dataset[idx]
        return raw_data


class SOLAMI(nn.Module):
    def __init__(
        self, 
        model_name_or_path,
        lora_model_name_or_path=None,
        output_dir=None,
        speech_tokenizer_path=None,
        speech_tokenizer_config=None,
        soundstorm_path=None,
        motion_tokenizer_dir=None,
        use_vllm=False,):
        super().__init__()
        available_gpu_ids = [i for i in range(torch.cuda.device_count()) if torch.cuda.is_available()]
        self.use_vllm = use_vllm
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.use_vllm:
            self.speech_device = torch.device('cuda:'+str(available_gpu_ids[1]))
            self.motion_device = torch.device('cuda:'+str(available_gpu_ids[1]))
        else:
            self.speech_device = torch.device('cuda:'+str(available_gpu_ids[1]))
            self.motion_device = torch.device('cuda:'+str(available_gpu_ids[0]))
        
        self.prompter = Prompter()
        print("loading speech tokenzier")
        
        
        self.speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)     
        self.speech_tokenizer.eval()
        self.speech_tokenizer.to(device=self.speech_device)
        self.soundstorm = load_soundstorm(soundstorm_path)
        self.soundstorm.eval()
        self.soundstorm.to(device=self.speech_device)
        
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
            getattr(self, key).to(device=self.motion_device)
        
        self.load_motion_utils()
        
        # model
        print("loading llm")
        if not self.use_vllm:
            if lora_model_name_or_path is None:
                self.model = LlamaForCausalLM.from_pretrained(
                    model_name_or_path,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    # device_map={"": self.device}
                    device_map="auto",
                    )
                self.model.half()
                print("Model loaded from {}".format(model_name_or_path))
            else:
                base_model = LlamaForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    )
                model = PeftModel.from_pretrained(base_model, lora_model_name_or_path, device_map="auto")
                self.model = model.merge_and_unload()
                print("LORA Model loaded from {}".format(lora_model_name_or_path))
            self.model.eval()
            if torch.__version__ >= "2" and sys.platform != "win32":
                self.model = torch.compile(self.model)
        else:
            from vllm import LLM, SamplingParams
                        
            self.llm_generator = LLM(model=model_name_or_path,
                                    tokenizer=model_name_or_path,
                                    tokenizer_mode='slow',
                                        tensor_parallel_size=1)
            ### slow mode vs fast mode!!!!!
            # self.sampling_params = SamplingParams(
            #     temperature=1.0,
            #     top_p=1.,
            #     max_tokens=600,
            #     min_tokens=10,
            #     repetition_penalty=1.0,
            #     )
            self.sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                max_tokens=600,
                min_tokens=10,
                repetition_penalty=1.0,
                )
            
        if CHECK_GPU_USAGE:
            self.model.to(self.device)
            
        
        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.session_id_to_conversations = {}
      
    
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

    
    def encode_speech(self, audio, sr=None):
        """
        input: audio file path or audio audio torch.tensor
        output: semantic token ids
        """
        if type(audio) is str:
            wav, sr = torchaudio.load(audio)
            # monophonic checking
            if wav.shape[0] > 1:
                wav = wav[:1, ]
        else:
            wav = audio
        if sr != self.speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).to(self.speech_device)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = self.speech_tokenizer.encode(wav) # codes: (n_q, B, T)
        return codes[0, 0, :]
    
    
    def decode_speech(self, content, prompt_path=None):
        """
        input: semantic codes list [[], ...]
        output: wav tensor
        """
        semantic_codes = content
        if prompt_path:
            # get tokens of prompt
            prompt_wav, sr = torchaudio.load(prompt_path)
            prompt_wav = prompt_wav.to(self.speech_device)
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
        # semantic_codes = [[int(num) for num in re.findall(r'\d+', content)]]
        # wav: (b, 1, t)
        config_dict = json.load(open('/root/pengyang/codebase/SOLAMI/models/vla/config/generate_config.json', 'r'))
        if type(semantic_codes) is list:
            semantic_codes = torch.tensor(semantic_codes).int().to(self.speech_device)
        wav = semantic2acoustic(semantic_codes, prompt_tokens, 
                                self.soundstorm, self.speech_tokenizer, steps=config_dict['vc_steps'])
        wav = wav.squeeze(0).detach().cpu()
        return wav


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
        del output, expression, betas, transl, global_orient, jaw_pose, leye_pose, reye_pose
        return joints[:, :, :55]


    def decode_motion(self, body_codes, hand_codes):
        if body_codes.shape[1] == 0:
            return None, None
        else:
            with torch.no_grad():
                body_repre = self.vae_body.decode(body_codes)
                hand_repre = self.vae_hand.decode(hand_codes)
            decode_repre = torch.cat((body_repre, hand_repre), dim=-1)
            mean = torch.tensor(self.motion_mean).to(decode_repre)
            std = torch.tensor(self.motion_std).to(decode_repre)
            decode_repre = decode_repre * std + mean
            smplx_params = recover_from_smplx_feature(decode_repre, 'local cont6d')
            smplx_joints = self.smplx_infer(smplx_params)
            return smplx_joints.squeeze(0).detach().cpu(), smplx_params.squeeze(0).detach().cpu()
    
    
    def llm_inference(
        self,
        input_ids,
    ):
        config_path='/root/pengyang/codebase/SOLAMI/models/vla/config/generate_config.json'
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
   
   
    def extract_LLM_part(self, text: str):
        """
        Extract the part of the text that is generated by the LLM.
        """
        matches = re.findall(r'(\[MMGPT\]|\[Human\]) : (.*?)(?=\s*\[|$)', text)
        conv_content_list = [{'role': role, 'content': content} for role, content in matches]
        return conv_content_list
    
    
    def get_code_from_str(self, input_str, re_format):
        _ids = re.findall(re_format, input_str)
        tmp_ids = [int(i) for i in _ids]
        output = torch.tensor(tmp_ids, dtype=int).reshape(1, -1)
        return output
    
    
    def create_session(self, character):
        """
        指定id!
        character: str,
        session_name: str,
        return: session id
        """
        chatbot_name = "[MMGPT]"
        user_name = "[Human]"
        user_end = "<eoh>"
        chatbot_end = "<eos>"
        eos_token = "<eos>"

        system_prompts = {
            'User': "You are a person of a 3D role-playing AI application",
            'Samantha': "You are an AI assistant named Samantha who can understand the human\'s body language, interact with human in real time, and perform sports, dance, and other skills with its body.",
            'Batman': "You are Batman, a superhero with superhuman strength, agility, and intelligence.",
            'Trump': "You are Donald Trump, the 45th President of the United States.",
            'Link': "You are Link, the main protagonist of The Legend of Zelda series.",
            'Banaya': "You are Bananya, a cat who lives inside a banana.",
            '11-45-G': "You are 11-45-G 11-45-G, a robot programmed to assist humans in their missions.",
        }
        conv = Conversation(
            name="SOLAMI",
            system_message=system_prompts,
            roles=(user_name, chatbot_name),
            messages=[],
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=user_end,
            sep2=chatbot_end
        )
        
        if character not in system_prompts:
            print(f"Character {character} is not supported.")
            return None
        
        voice_dir = "SOLAMI_data/audio/voice_prompt/voices"
        voice_prompts = {
            'User': ["User_0.wav", "User_1.wav", "User_2.wav"],
            'Samantha': ["Samantha_0.wav", "Samantha_1.wav", "Samantha_2.wav"],
            'Batman': ["batman_0.mp3", "batman_1.mp3"],
            'Trump': ["trump_0.mp3", "trump_1.mp3"],
            'Link': ["Link_0.wav", "Link_1.wav", "Link_2.wav"],
            'Banaya': ["Banaya_0.wav", "Banaya_1.wav", "Banaya_2.wav"],
            '11-45-G': ["11-45-G_0.wav", "11-45-G_1.wav", "11-45-G_2.wav"],
        }
        ## random choice a prompt
        character_voice_prompt = random.choice(voice_prompts[character])
        agent_prompt_path = os.path.join(voice_dir, character_voice_prompt)
        user_voice_prompt = random.choice(voice_prompts['User'])
        user_prompt_path = os.path.join(voice_dir, user_voice_prompt)
        conv.user_voice_prompt = user_prompt_path
        conv.agent_voice_prompt = agent_prompt_path
        
        for i in range(10000):
            session_id = character + '_' + str(i)
            if session_id not in self.session_id_to_conversations:
                self.session_id_to_conversations[session_id] = conv
                return session_id
   
    
    def conv_inference_using_modality_ids(self, motion, speech, session_id):
        
        """
        input motion path & speech path
        output motion & speech ids
        motion: dict of motion codes
        speech: speech codes
        session_id: str
        """
        user_name_ = "[Human]"
        chatbot_name_ = "[MMGPT]"
        
        latency_cost = {}
        
        prepare_time = time.time()
        
        if session_id not in self.session_id_to_conversations:
            print(f"Session {session_id} does not exist.")
            return None
        conv = self.session_id_to_conversations[session_id]

        ### get motion & speech 
        
        speech_ids = speech
        speech_str = modality_tokens_to_string(speech_ids, modality="speech")
        
        body_ids = motion['body']
        hand_ids = motion['hand']
        trans_ids = motion['trans']
        motion_str = modality_tokens_to_string(trans_ids, modality="trans") +\
                        modality_tokens_to_string(body_ids, modality="body") +\
                            modality_tokens_to_string(hand_ids, modality="hand")
                        
        ### get input string
        message_str = motion_str + speech_str
        conv.append_message(user_name_, message_str)
        agent = self.get_character_from_session(session_id)
        
        ### check length & get input token ids
        input_ids = torch.zeros((1, 1))
        start_rounds = 0
        while input_ids.shape[1] == 1 or input_ids.shape[1] >= LLM_MAX_LENGTH - 600:
            input_conv_str = conv.get_prompt(agent_role=agent, start_rounds=start_rounds)
            if input_conv_str == "":
                print("Conversation is not valid.")
                return None
            input_ids = self.tokenizer(
                input_conv_str,
                truncation=True,
                max_length=LLM_MAX_LENGTH,
                padding=True,
                return_tensors='pt',
            ).input_ids
            start_rounds += 2
        
        
        # print('input_ids length', input_ids.shape)
        prepare_time = time.time() - prepare_time
        latency_cost['prepare_time'] = prepare_time
        
        for repeat in range(REPEAT_TIMES):
            ### LLM inference
            llm_time = time.time()
            if self.use_vllm:
                outputs = self.llm_generator.generate(input_conv_str, self.sampling_params)
                llm_response = outputs[0].outputs[0].text
                response_part = {'role': chatbot_name_, 'content': llm_response}
                llm_time = time.time() - llm_time
                latency_cost['llm_time'] = llm_time
            else:
                input_ids = input_ids.to(self.device)
                llm_response = self.llm_inference(input_ids)
                
                llm_time = time.time() - llm_time
                latency_cost['llm_time'] = llm_time
                ### check repeat or batch generation TODO
                torch.cuda.empty_cache()
                ### extract motion & speech ids & update conversation
                
                conv_content_list = self.extract_LLM_part(llm_response)
                response_part = conv_content_list[-1]
            if response_part['role'] != chatbot_name_:
                print("Response is not valid. & Repeat.", repeat)
                continue
                # return None 
            else:
                response_str = response_part['content']
                response_str_dict = {}
                ### for motion
                for modality in ['body', 'hand']:
                    special_dict = modal_special_str[modality]
                    modality_content = extract_text_between_tags(response_str, tag1=special_dict['sos'], tag2=special_dict['eos'])
                    if modality_content:
                        response_str_dict[modality] = modality_content
                    else:
                        response_str_dict[modality] = None
                ### for speech
                speech_special_str = modal_special_str['speech']
                speech_content = extract_text_between_tags(response_str, tag1=speech_special_str['sos'], tag2=speech_special_str['eos'])
                if speech_content:
                    response_str_dict['speech'] = speech_content
                else:
                    response_str_dict['speech'] = None
                
                conv.append_message(chatbot_name_, response_str)
                
                ### decode motion & speech
                body_ids = self.get_code_from_str(response_str_dict['body'], r'<🕺(\d+)>')
                hand_ids = self.get_code_from_str(response_str_dict['hand'], r'<🤚(\d+)>')
                min_len = min(body_ids.shape[1], hand_ids.shape[1])
                body_ids = body_ids[:, :min_len]
                hand_ids = hand_ids[:, :min_len]
                speech_ids = self.get_code_from_str(response_str_dict['speech'], r'<🗣️(\d+)>')
                if body_ids.shape[1] == 0 or speech_ids.shape[1] == 0:
                    print("Response check motion and speech is not valid. & Repeat.", repeat)
                    continue
                    # return None
                
                motion_decode_time = time.time()
                smplx_joints, smplx_params = self.decode_motion(body_ids.to(self.motion_device), hand_ids.to(self.motion_device))
                motion_decode_time = time.time() - motion_decode_time
                latency_cost['motion_decode_time'] = motion_decode_time
                
                speech_decode_time = time.time()
                response_speech = self.decode_speech(speech_ids.to(self.speech_device), conv.agent_voice_prompt)
                speech_decode_time = time.time() - speech_decode_time
                latency_cost['speech_decode_time'] = speech_decode_time
                
                ### decide return results
                
                return {
                    'motion_joints': smplx_joints,
                    'speech': response_speech,
                    'motion_params': smplx_params,
                    'latency': latency_cost,
                }
    
    
    def conv_delete_session(self, session_id):
        if session_id in self.session_id_to_conversations:
            del self.session_id_to_conversations[session_id]
        else:
            print(f"Session {session_id} does not exist.")


    def conv_reset_session(self, session_id):
        if session_id in self.session_id_to_conversations:
            self.session_id_to_conversations[session_id].reset()
        else:
            print(f"Session {session_id} does not exist.")
    
    
    def get_available_sessions(self, ):
        return list(self.session_id_to_conversations.keys())
    
    
    def get_character_from_session(self, session_id):
        if session_id in self.session_id_to_conversations:
            return session_id.split('_')[0]
        else:
            print(f"Session {session_id} does not exist.")
            return None

    

if __name__ == "__main__":
    # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    # initialize_distributed()
    # initialize_debugpy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--period", type=int, default=4)
    parser.add_argument("--model_name_or_path", type=str, 
                        default="/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain_audio_motion_final/checkpoint-2560")
    parser.add_argument("--lora_model_name_or_path", type=str,
                        default=None)
    parser.add_argument("--output_dir", type=str,
                        default="/root/pengyang/codebase/SOLAMI/models/vla/infer_output/conv_inference")
    parser.add_argument('--use_vllm', type=bool, default=False)
    args = parser.parse_args()

    # print('check bash script')
    # print(args.output_dir)
    # exit()

    if args.lora_model_name_or_path is not None:
        LORA = True
        model_name_or_path = args.model_name_or_path
        lora_model_name_or_path = args.lora_model_name_or_path
        # model_name_or_path = "/root/pengyang/codebase/SOLAMI/models/vla/output_models/pretrain_audio_motion_final/checkpoint-2560"
        # lora_model_name_or_path = "/root/pengyang/codebase/SOLAMI/models/vla/output_models/it_lora_deepspeed/checkpoint-640"
    else:
        LORA = False
        model_name_or_path = args.model_name_or_path
        lora_model_name_or_path = None
        # model_name_or_path = "/root/pengyang/codebase/SOLAMI/models/vla/output_models/it_full_deepspeed_multinode/checkpoint-384"
        # lora_model_name_or_path = None
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    speech_tokenizer_path = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/ckpt.dev"
    speech_tokenizer_config = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/config.json"
    soundstorm_path = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/soundstorm/speechtokenizer_soundstorm_mls.pt"
    motion_tokenizer_dir = "/root/pengyang/codebase/SOLAMI/extra/motion_tokenizer_final"

    conv_model = SOLAMI(model_name_or_path, lora_model_name_or_path, 
                        output_dir, speech_tokenizer_path, 
                        speech_tokenizer_config, soundstorm_path, 
                        motion_tokenizer_dir, args.use_vllm)

    test_dataset = ITDataset("SOLAMI_data/Conversation/test_it_items.jsonl")

    for i in tqdm(range(args.part, len(test_dataset), args.period)):
        data_item = test_dataset[i]
        data_idx = data_item['id']
        
        save_path = os.path.join(output_dir, data_idx+'.npz')
        if os.path.exists(save_path):
            continue
        save_res = {}
        latency_res = {}
        
        raw_chat_data = data_item['chat']
        character = raw_chat_data[1]['role']
        session_id = conv_model.create_session(character)
        if session_id is None:
            print(f"Session {session_id} is not created at {data_idx}.")
            continue
        for j, chat_round in enumerate(raw_chat_data):
            if chat_round['role'] != character:
                body = chat_round['body']
                hand = chat_round['hand']
                trans = chat_round['trans']
                motion = {'body': body, 'hand': hand, 'trans': trans}
                speech = chat_round['speech']
                response_res = conv_model.conv_inference_using_modality_ids(motion, speech, session_id)
                if response_res is None:
                    print(f"Session {session_id} round {j} is not valid at {data_idx}.")
                    break
                else:
                    save_res[str(j)] = {}
                    for key in [ 'motion_params', 'speech']:
                        save_res[str(j)][key] = response_res[key].numpy()
                    latency_res[str(j)] = {'latency': response_res['latency']}
            else:
                continue
        conv_model.conv_delete_session(session_id)
        ### save res
        np.savez(save_path, **save_res)
        latency_save_dir = os.path.join(output_dir, 'latency', data_idx)
        os.makedirs(latency_save_dir, exist_ok=True)
        with open(os.path.join(latency_save_dir, 'speech_text.json'), 'w', encoding='utf-8') as f:
            json.dump(latency_res, f, indent=4, ensure_ascii=False)
        torch.cuda.empty_cache()
        pass
    print("All Done!")