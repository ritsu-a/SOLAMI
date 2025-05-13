import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla')
sys.path.append('SOLAMI')
sys.path.append("/root/pengyang/codebase/SOLAMI/models/motiongpt")
import torch
import torch.nn as nn
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass, field
from typing import Optional
import argparse
import json
import re
import torchaudio
from einops import rearrange
import yaml
from tqdm import tqdm
sys.path.append('tools/smplx')
import smplx
import time
import whisper
from omegaconf import OmegaConf
import asyncio



import torch.distributed as dist
import debugpy

def initialize_debugpy():
    # if not dist.is_initialized() or dist.get_rank() == 0:
        # print(f"Rank: {dist.get_rank()} - Debugpy is listening on port 15696")
        print("Debugpy is listening on port 15697")
        debugpy.listen(("0.0.0.0", 15697))
        debugpy.wait_for_client()
        
# initialize_debugpy()

from vllm import LLM, SamplingParams

from TTS.api import TTS

from FlagEmbedding import FlagModel

from torch.utils.data import Dataset as DatasetTorch
from datasets import load_dataset
from transformers import  LlamaForCausalLM, LlamaTokenizer, GenerationConfig, EncodecModel, AutoProcessor
from m_utils.prompter import *
from infer.pre_post_process import extract_text_between_tags
from motion.smplx_process import recover_from_smplx_feature, process_smplx_feature, preprocess_smplx
from motion.motion_dataset import MotionTextDataset
from motion.unified_dataset import UnifiedDataset

from mGPT.config import instantiate_from_config
from mGPT.config import instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.utils.load_checkpoint import load_pretrained

from m_utils.loggings import get_logger
from m_utils.prompter import *
from m_utils.conv import Conversation, SeparatorStyle


DEBUG=False

def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data



class ITDataset(DatasetTorch):
    def __init__(self, 
                data_path: str,
                raw_dataset=None,
                raw_motion_dataset=None,
                method='llm+speech'):
        super(ITDataset, self).__init__()
        if raw_dataset is not None:
            self.raw_dataset = raw_dataset
        else:
            print("Loading from dataset {}".format(data_path.split('/')[-1]))
            raw_dataset = load_dataset("json", data_files=data_path)
            raw_dataset = raw_dataset['train']
            if DEBUG:
                LENGTHS=180
                raw_dataset = raw_dataset.select(range(LENGTHS))
            self.raw_dataset = raw_dataset
        print("Loading {} items from dataset {}".format(len(self.raw_dataset), data_path.split('/')[-1]))
        self.raw_motion_dataset = raw_motion_dataset
        

    def __len__(self):
        return len(self.raw_dataset)


    def __getitem__(self, idx):
        raw_data = self.raw_dataset[idx]
        return raw_data


class DLP(nn.Module):
    def __init__(
        self, 
        model_path="/root/pengyang/codebase/SOLAMI/extra/meta-llama/Llama-2-7b-chat-hf",
        REPEAT_TIMES=4, ):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.REPEAT_TIMES = REPEAT_TIMES
        self.prompter = Prompter()
        self.llm_generator = LLM(model=model_path)
        self.sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                max_tokens=150,
                presence_penalty=1.0,
        )
        ### TTS 
        self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.DEVICE)
        
        ### ASR
        self.whisper_model = whisper.load_model('large-v3', download_root='~/.cache/whisper', device=self.DEVICE)
        
        ### motion
        # if self.method in ['dlp+retrieval', 'dlp+motiongpt+retrieval']:
        ### text embedding retrieval
        self.text_embedding_model = FlagModel('BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            query_instruction_format="{}{}",
            use_fp16=True,
            devices=str(self.DEVICE),   # if you don't have a GPU, you can use "cpu"
            pooling_method='cls',)
        
        
        # loading motiongpt
        # if self.method in ['dlp+motiongpt', 'dlp+motiongpt+retrieval']:
        cfg = OmegaConf.load("/root/pengyang/codebase/SOLAMI/models/motiongpt/experiments/mgpt/Pretrain_HumanML3D_GPT2_Local_Body_Hand_Sep_NoInterleave/config_2024-09-24-20-26-38_train.yaml")
        cfg.TEST.CHECKPOINTS = "/root/pengyang/codebase/SOLAMI/models/motiongpt/experiments/mgpt/Pretrain_HumanML3D_GPT2_Local_Body_Hand_Sep_NoInterleave/checkpoints/last.ckpt"
        cfg.TEST.BATCH_SIZE = 1
        cfg.EXPER.transform = False
        cfg.DEBUG = True
        cfg.METRIC.TM2T.t2m_path = "/root/pengyang/codebase/SOLAMI/models/motiongpt/deps/t2m"
        cfg.METRIC.TYPE = []
        cfg.lm.gpt2_medium.params.model_path = "/root/pengyang/codebase/SOLAMI/models/motiongpt/deps/gpt2"
        datasets_motiongpt = build_data(cfg, phase='token')
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        model_config['params']['cfg'] = cfg
        model_config['params']['datamodule'] = datasets_motiongpt
        self.motiongpt_model = instantiate_from_config(model_config)
        load_pretrained(cfg, self.motiongpt_model, phase="test", strict=False)
        self.motiongpt_model = self.motiongpt_model.to(self.DEVICE)
        self.motiongpt_model.eval()
        
        self.load_motion_utils()
        self.load_motion_datasets()
            
        self.session_id_to_conversations = {}
        self.inference_first_round()
      
    
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
        
        return self.motion_mean, self.motion_std, self.transform_mean_part, self.transform_std_part, self.betas, self.t_root_J


    def load_motion_datasets(self):
        motion_mean, motion_std, transform_mean, transform_std, betas, t_root_J = self.load_motion_utils()

        self.raw_motion_dataset = MotionTextDataset(
            mean=motion_mean,
            std=motion_std,
            transform_mean=transform_mean,
            transform_std=transform_std,
            tmpFile=True,
            tiny=False,
        )
    
        unified_configs = {
                'dlp': {
                    'text_embeddings': "SOLAMI_data/DLP-MoCap/embeddings/bge_large_embeddings.npz",
                    'dataset_items': "SOLAMI_data/DLP-MoCap/dataset_items_post.json",
                    'dataset_root_dir': "SOLAMI_data/DLP-MoCap",
                },
                'humanml3d': {
                    'text_embeddings': "SOLAMI_data/HumanML3D/embeddings/bge_large_embeddings.npz",
                    'dataset_items': "SOLAMI_data/HumanML3D/dataset_items_post.json",
                    'dataset_root_dir': "SOLAMI_data/HumanML3D",
                },
                'inter-x': {
                    'text_embeddings': "SOLAMI_data/Inter-X/embeddings/bge_large_embeddings.npz",
                    'dataset_items': "SOLAMI_data/Inter-X/dataset_items_post.json",
                    'dataset_root_dir': "SOLAMI_data/Inter-X",
                },
            }
        
        ### DEBUG
        # unified_configs.pop('humanml3d')
        # unified_configs.pop('inter-x')
        unified_configs.pop('dlp')
        # unified_configs.pop('inter-x')
        self.unified_dataset = UnifiedDataset(unified_configs, device=self.DEVICE)


    def generate_motion(self, input_text):
        repeat_samples = 3
        outputs = self.motiongpt_model.lm.generate_conditional([input_text] * repeat_samples, task="t2m")
        for i in range(len(outputs)):
            outputs[i] = torch.clamp(outputs[i], 0, 512 - 1, out=None)
            assert len(outputs[i]) == 2
            if len(outputs[i][0]) > 1:
                body_feat = self.motiongpt_model.vae_body.decode(outputs[i][0])
                hand_feat = self.motiongpt_model.vae_hand.decode(outputs[i][1])
                motion = torch.cat((body_feat, hand_feat), dim=-1)
                break
            else:
                motion = None
        if motion is not None:
            motion_params = self.motiongpt_model.feats2joints(motion)
            motion_params = motion_params.squeeze(0).cpu().numpy()
        else:
            motion_params = None

        return motion_params


    def motion_caption_infer(self, input_motion_path):
        if input_motion_path is None:
            return ''
        
        loaded_motion = np.load(input_motion_path, allow_pickle=True)
        motion_dict = dict(loaded_motion)

        pose_index = list(range(0, 66)) + list(range(75, 165))
        motion = np.concatenate([motion_dict['trans'], motion_dict['poses'][:, pose_index]], axis=1)
        
        ### 159 feature motion input
        motion = torch.tensor(motion, dtype=torch.float32).to(self.DEVICE)
        if len(motion.shape) == 2:
            motion = motion.unsqueeze(0)
        
        motion_joints, motion_t_root_J = self.smplx_infer(motion)
        new_root_rotvec, new_transl, (smplx_quat, smplx_root) = preprocess_smplx(motion_joints, motion[..., :3], motion[..., 3:6], motion_t_root_J)
        new_motion = torch.cat([new_transl, new_root_rotvec, motion[..., 6:]], dim=-1)    
        
        motion_feature = process_smplx_feature(new_motion)
        mean = torch.tensor(self.motion_mean, dtype=torch.float32).to(self.DEVICE)
        std = torch.tensor(self.motion_std, dtype=torch.float32).to(self.DEVICE)
        motion_feature_input = (motion_feature - mean) / std
        
        code_pred_body, _ = self.motiongpt_model.vae_body.encode(motion_feature_input[..., :self.motiongpt_model.vae_body.nfeats])
        code_pred_hand, _ = self.motiongpt_model.vae_hand.encode(motion_feature_input[..., self.motiongpt_model.vae_body.nfeats:])
        code_pred = torch.cat([code_pred_body, code_pred_hand], dim=0)
        
        motion_tokens = [code_pred] * 3
        lengths_tokens = [len(code_pred[0])] * 3
        outputs = self.motiongpt_model.lm.generate_conditional(motion_tokens=motion_tokens,
                                                                lengths=lengths_tokens,
                                                                task="m2t",
                                                                stage='test')
        for caption in outputs:
            if caption.strip() != '':
                return caption
        return ''


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
        t_root_J = output.t_root_J.clone()
        del output, expression, betas, transl, global_orient, jaw_pose, leye_pose, reye_pose
        return joints[:, :, :55], t_root_J

    
    def create_session(self, character, session_id, method='llm+speech'):
        """
        character: str,
        session_name: str,
        return: session id
        """
        
        if session_id in self.session_id_to_conversations:
            print(f"Session {session_id} already exists.")
            return None
        
        chatbot_name = "[MMGPT]"
        user_name = "[Human]"
        user_end = "<eoh>"
        chatbot_end = "<eos>"
        eos_token = "<eos>"

        system_prompts = {
            'User': "You are a person of a 3D role-playing AI application",
            'Samantha': "You are an AI assistant named Samantha who can understand the human\'s body language and perform sports, dance, and other skills with its body.",
            'Batman': "You are Batman, a superhero with superhuman strength, agility, and intelligence.",
            'Trump': "You are Donald Trump, the 45th President of the United States.",
            'Link': "You are Link, the main protagonist of The Legend of Zelda series.",
            'Banaya': "You are Bananya, a cat who lives inside a banana.",
            '11-45-G': "You are 11-45-G 11-45-G, a robot programmed to assist humans in their missions.",
        }
        
        start_prompt = "Let's do a role play! "
        if method == 'llm+speech':
            format_prompt = "Let's talk in format:   <speech>XXXX</speech> \n <speech>XXXX</speech> is just the words of the character, no motion or expression, at most 15 words. For example, my message is: <speech>Hello, how are you recently?</speech>  \n Remember you must response in this format. Let's start."
        elif method in ['dlp+retrieval', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']:
            format_prompt = "Let's behave in format:   <speech>XXXX</speech> <motion>YYYY</motion> \n <speech>XXXX</speech> means the words of the character, at most 15 words. <motion>YYYY</motion> means the text description of the character, no emojis. "
            format_prompt += "For example, my message is: <speech>Hello, how are you recently?</speech> <motion>Wave hands</motion>  \n Remember you must response in this format. Let's start."
        else:
            print('Invalid method ', method)
            return None
        
        for key in system_prompts:
            system_prompts[key] = start_prompt + system_prompts[key] + format_prompt
        
        conv = Conversation(
            name="SOLAMI",
            system_message=system_prompts,
            roles=(user_name, chatbot_name),
            character=character,
            messages=[],
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=user_end,
            sep2=chatbot_end,
            method=method,
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
        
        self.session_id_to_conversations[session_id] = conv
        print(f"Session {session_id} with {method} created.")
        print(f"Character {character} is ready to interact.")
   
   
    def characters(self):
        characters = ['User', 'Samantha', 'Batman', 'Trump', 'Link', 'Banaya', '11-45-G',]
        return characters
    
    def get_response_count(self, session_id):
        if session_id not in self.session_id_to_conversations:
            print(f"Session {session_id} does not exist.")
            return None
        conv = self.session_id_to_conversations[session_id]
        return conv.get_message_rounds()
    
    
    def inference_methods(self):
        return ['llm+speech', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']
    
    def inference_first_round(self,):
        input_motion_path = "/root/pengyang/codebase/SOLAMI/models/vla/data/test_server/motion_user.npz"
        input_audio_path = "SOLAMI_data/audio/voice_prompt/voices/trump_0.mp3"
        
        output_motion_path = "/root/pengyang/codebase/SOLAMI/models/vla/data/test_server/output_character1_1.npz"
        output_audio_path = "/root/pengyang/codebase/SOLAMI/models/vla/data/test_server/output_character1_1.mp3"
        session_id = self.create_session('Trump', 'Trump_0', 'dlp+motiongpt+retrieval')
        res = self.conv_inference_server_local(input_motion_path, input_audio_path, 
                                         output_motion_path=output_motion_path, 
                                         output_audio_path=output_audio_path, 
                                         session_id='Trump_0')
        self.conv_delete_session('Trump_0')
        pass
    
   
    def clean_speech_text_response(self, text_response):
        ### extract part after <speech>
        ## remove none ascii code
        cleaned_text_response = re.sub(r'[^\x00-\x7F]+', '', text_response)
        speech_content = re.search(r"<speech>(.*?)</speech>", cleaned_text_response)
        if speech_content:
            cleaned_text = re.sub(r"\*.*?\*", "", speech_content.group(1))
            return cleaned_text
        else:
            return ''
   
   
    def clean_speech_motion_text_response(self, text_response):
        ### extract part after <speech>
        cleaned_text_response = re.sub(r'[^\x00-\x7F]+', '', text_response)
        
        speech_content = re.search(r"<speech>(.*?)</speech>", cleaned_text_response)
        if speech_content:
            cleaned_text = re.sub(r"\*.*?\*", "", speech_content.group(1))
        else:
            cleaned_text = ''
        motion_content = re.search(r"<motion>(.*?)</motion>", cleaned_text_response)
        if motion_content:
            motion_text = re.sub(r"\*.*?\*", "", motion_content.group(1))
        else:
            motion_text = ''
        return cleaned_text, motion_text
        
    
    def shorten_text_response(self, text, max_words=20, max_sentences=3):
        if text.strip() == '':
            return ''
        
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        
        for j in range(max_sentences, 0, -1):
            selected_sentences = sentences[:j]
            shortened_text = ' '.join(selected_sentences)
            words = shortened_text.split()
            if len(words) > max_words:
                continue
            return shortened_text
        
        return sentences[0]
    
    
    def conv_inference(self, motion, speech, session_id, save_dir):
        
        user_name_ = "user"
        chatbot_name_ = "assistant"
        
        latency_cost = {}
        
        input_time = time.time()
        
        if session_id not in self.session_id_to_conversations:
            print(f"Session {session_id} does not exist.")
            return None
        conv = self.session_id_to_conversations[session_id]

        method = conv.method

        ### get motion & speech 
        if type(speech) is str:
            if not os.path.exists(speech):
                speech_str = ''
            else:
                speech_str = self.whisper_model.transcribe(speech)['text']
        elif speech is None:
            speech_str = ''
        else:
            speech_str = self.whisper_model.transcribe(speech)['text']
        
        if method in ['dlp+motiongpt', 'dlp+motiongpt+retrieval']:
            motion_caption = self.motion_caption_infer(motion)
            motion_str = motion_caption
        else:
            motion_str = ''
        
        if method == 'llm+speech':
            message_str = '<speech>' + speech_str +'</speech>'
        elif method in ['dlp+retrieval', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']:
            message_str = '<speech>' + speech_str +'</speech> <motion>' + motion_str + '</motion>'
        else:
            print('Invalid method ', method)
        ### get input string
        conv.append_message(user_name_, message_str)
        agent = self.get_character_from_session(session_id)
        
        # print('input_ids length', input_ids.shape)
        input_time = time.time() - input_time
        latency_cost['input_time'] = input_time
        
        for repeat in range(self.REPEAT_TIMES):
            ### LLM inference
            llm_time = time.time()
            start_rounds = max(0, conv.get_message_rounds()-5)
            conversations = conv.to_llama_chat(start_rounds)
            llm_response = self.llm_generator.chat(messages=conversations,
                                    sampling_params=self.sampling_params,
                                    use_tqdm=False)
            
            text_response = llm_response[0].outputs[0].text
            
            llm_time = time.time() - llm_time
            latency_cost['llm_time'] = llm_time
            ### check repeat or batch generation 
            torch.cuda.empty_cache()
            ### extract motion & speech ids & update conversation
            
            gen_time = time.time()
            
            if method == 'llm+speech':
                cleaned_speech_response = self.clean_speech_text_response(text_response)
                cleaned_speech_response = self.shorten_text_response(cleaned_speech_response)
            else:
                cleaned_speech_response, cleaned_motion_response = self.clean_speech_motion_text_response(text_response)
                cleaned_speech_response = self.shorten_text_response(cleaned_speech_response)
                pass
                
            if method == 'llm+speech':
                if cleaned_speech_response:
                    conv.append_message(chatbot_name_, '<speech>' + cleaned_speech_response +'</speech>')
                    save_path = os.path.join(save_dir, f"pred_{conv.get_message_rounds()-1}.wav")
                    self.xtts_model.tts_to_file(text=cleaned_speech_response, speaker_wav=conv.agent_voice_prompt, language="en", file_path=save_path)
                    gen_time = time.time() - gen_time
                    latency_cost['gen_time'] = gen_time
                    return {
                        'speech_text': cleaned_speech_response,
                        'latency': latency_cost,
                    }
            elif method in ['dlp+retrieval', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']:
                if cleaned_speech_response and cleaned_motion_response:
                    conv.append_message(chatbot_name_, '<speech>' + cleaned_speech_response +'</speech> <motion>' + cleaned_motion_response + '</motion>')
                    save_path = os.path.join(save_dir, f"pred_{conv.get_message_rounds()-1}.wav")
                    self.xtts_model.tts_to_file(text=cleaned_speech_response, speaker_wav=conv.agent_voice_prompt, language="en", file_path=save_path)
                    
                    if method in ['dlp+retrieval', 'dlp+motiongpt+retrieval']:
                        motion_query_embedding = self.text_embedding_model.encode_queries([cleaned_motion_response])
                        re_actions, re_motions = self.unified_dataset.retrieval_motion_by_embeddings(motion_query_embedding[0], top_k=5)
                        gen_motion = None
                        for motion_id in re_motions:
                            gen_motion = self.raw_motion_dataset.get_data_from_name(motion_id)[1]
                            if gen_motion is not None:
                                break
                    else:
                        gen_motion = self.generate_motion(cleaned_motion_response)
                    
                    gen_time = time.time() - gen_time
                    latency_cost['gen_time'] = gen_time
                    
                    if gen_motion is not None:
                        return {
                            'speech_text': cleaned_speech_response,
                            'latency': latency_cost,
                            'motion_params': gen_motion,
                        }
                    pass
    
        return None


    
    async def _get_speech_text_input(self, input_audio_path):
        if not os.path.exists(input_audio_path):
            return ''
        else:
            return self.whisper_model.transcribe(input_audio_path)['text']


    async def _get_motion_text_input(self, input_motion_path, method):
        if method in ['dlp+motiongpt', 'dlp+motiongpt+retrieval']:
            return self.motion_caption_infer(input_motion_path)
        else:
            return ''
    
    
    async def _tts(self, text, speaker_wav, output_audio_path):
        await asyncio.to_thread(
            self.xtts_model.tts_to_file,
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=output_audio_path
        )
    
    
    async def _generate_motion_from_text(self, method, text):
        if method in ['dlp+retrieval', 'dlp+motiongpt+retrieval']:
            motion_query_embedding = self.text_embedding_model.encode_queries([text])
            re_actions, re_motions = self.unified_dataset.retrieval_motion_by_embeddings(motion_query_embedding[0], top_k=5)
            gen_motion = None
            for motion_id in re_motions:
                gen_motion = self.raw_motion_dataset.get_data_from_name(motion_id)[1]
                if gen_motion is not None:
                    break
        else:
            gen_motion = self.generate_motion(text)
        return gen_motion
        
    
    async def _conv_inference_server(self,  
                            input_motion_path, 
                            input_audio_path, 
                            output_motion_path, 
                            output_audio_path, 
                            session_id):
        
        user_name_ = "user"
        chatbot_name_ = "assistant"
        
        latency_cost = {}
        start_time = time.time()
        input_time = time.time()
        
        if session_id not in self.session_id_to_conversations:
            print(f"Session {session_id} does not exist.")
            return None
        conv = self.session_id_to_conversations[session_id]

        method = conv.method

        speech_task = self._get_speech_text_input(input_audio_path)
        motion_task = self._get_motion_text_input(input_motion_path, method)
        
        speech_str, motion_str = await asyncio.gather(speech_task, motion_task)
        
        if method == 'llm+speech':
            message_str = '<speech>' + speech_str +'</speech>'
        elif method in ['dlp+retrieval', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']:
            message_str = '<speech>' + speech_str +'</speech> <motion>' + motion_str + '</motion>'
        else:
            print('Invalid method ', method)
        ### get input string
        conv.append_message(user_name_, message_str)
        agent = self.get_character_from_session(session_id)
        
        # print('input_ids length', input_ids.shape)
        input_time = time.time() - input_time
        latency_cost['input_time'] = input_time
        
        for repeat in range(self.REPEAT_TIMES):
            ### LLM inference
            llm_time = time.time()
            start_rounds = max(0, conv.get_message_rounds()-5)
            conversations = conv.to_llama_chat(start_rounds)
            llm_response = self.llm_generator.chat(messages=conversations,
                                    sampling_params=self.sampling_params,
                                    use_tqdm=False)
            
            text_response = llm_response[0].outputs[0].text
            
            llm_time = time.time() - llm_time
            latency_cost['llm_time'] = llm_time
            ### check repeat or batch generation 
            torch.cuda.empty_cache()
            ### extract motion & speech ids & update conversation
            
            gen_time = time.time()
            
            if method == 'llm+speech':
                cleaned_speech_response = self.clean_speech_text_response(text_response)
                cleaned_speech_response = self.shorten_text_response(cleaned_speech_response)
            else:
                cleaned_speech_response, cleaned_motion_response = self.clean_speech_motion_text_response(text_response)
                cleaned_speech_response = self.shorten_text_response(cleaned_speech_response)
                pass
                
            if method == 'llm+speech':
                if cleaned_speech_response:
                    conv.append_message(chatbot_name_, '<speech>' + cleaned_speech_response +'</speech>')
                    # save_path = os.path.join(save_dir, f"pred_{conv.get_message_rounds()-1}.wav")
                    self.xtts_model.tts_to_file(text=cleaned_speech_response, speaker_wav=conv.agent_voice_prompt, language="en", file_path=output_audio_path)
                    gen_time = time.time() - gen_time
                    latency_cost['gen_time'] = gen_time
                    overall_time = time.time() - start_time
                    latency_cost['overall_time'] = overall_time
                    return {
                        'speech_text': cleaned_speech_response,
                        'latency': latency_cost,
                    }
            elif method in ['dlp+retrieval', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']:
                if cleaned_speech_response and cleaned_motion_response:
                    conv.append_message(chatbot_name_, '<speech>' + cleaned_speech_response +'</speech> <motion>' + cleaned_motion_response + '</motion>')
                    
                    tts_task = self._tts(cleaned_speech_response, conv.agent_voice_prompt, output_audio_path)
                    motion_task = self._generate_motion_from_text(method, cleaned_motion_response)
                    gen_motion = await motion_task
                    await tts_task
                    
                    gen_time = time.time() - gen_time
                    latency_cost['gen_time'] = gen_time
                    
                    if gen_motion is not None:
                        
                        trans = gen_motion[..., :3]
                        poses = gen_motion[..., 3:].reshape(-1, 52, 3)
                        poses = np.concatenate([poses[..., :22, :], np.zeros((poses.shape[0], 3, 3)), poses[..., 22:, :]], axis=1)
                        poses = poses.reshape(-1, 165)
                        np.savez(output_motion_path, trans=trans, poses=poses)
                        
                        overall_time = time.time() - start_time
                        latency_cost['overall_time'] = overall_time
                        
                        return {
                            'speech_text': cleaned_speech_response,
                            'latency': latency_cost,
                            'motion_params': gen_motion,
                        }
                    pass
    
        return None


    def print_info(self,input_motion_path, 
                       input_audio_path, 
                       output_motion_path, 
                       output_audio_path, ):
        print('input_motion_path', input_motion_path)
        print('input_audio_path', input_audio_path)
        print('output_motion_path', output_motion_path)
        print('output_audio_path', output_audio_path)
        

    def conv_inference_server(self,  
                            input_motion_path, 
                            input_audio_path, 
                            output_motion_path, 
                            output_audio_path, 
                            session_id):
        
        user_name_ = "user"
        chatbot_name_ = "assistant"
        
        latency_cost = {}
        start_time = time.time()
        input_time = time.time()
        
        self.print_info(
            input_motion_path, 
            input_audio_path, 
            output_motion_path, 
            output_audio_path,
        )
        
        
        if session_id not in self.session_id_to_conversations:
            print(f"Session {session_id} does not exist.")
            return None
        conv = self.session_id_to_conversations[session_id]

        method = conv.method
        
        ## get motion & speech 
        if not os.path.exists(input_audio_path):
            speech_str = ''
        else:
            speech_str = self.whisper_model.transcribe(input_audio_path)['text']
        
        if method in ['dlp+motiongpt', 'dlp+motiongpt+retrieval']:
            motion_caption = self.motion_caption_infer(input_motion_path)
            motion_str = motion_caption
        else:
            motion_str = ''
        # speech_task = self._get_speech_text_input(input_audio_path)
        # motion_task = self._get_motion_text_input(input_motion_path, method)
        
        # speech_str, motion_str = await asyncio.gather(speech_task, motion_task)
        
        if method == 'llm+speech':
            message_str = '<speech>' + speech_str +'</speech>'
        elif method in ['dlp+retrieval', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']:
            message_str = '<speech>' + speech_str +'</speech> <motion>' + motion_str + '</motion>'
        else:
            print('Invalid method ', method)
        ### get input string
        conv.append_message(user_name_, message_str)
        agent = self.get_character_from_session(session_id)
        
        print('input message_str', message_str)
        
        # print('input_ids length', input_ids.shape)
        input_time = time.time() - input_time
        latency_cost['input_time'] = input_time
        
        for repeat in range(self.REPEAT_TIMES):
            ### LLM inference
            llm_time = time.time()
            start_rounds = max(0, conv.get_message_rounds()-5)
            conversations = conv.to_llama_chat(start_rounds)
            llm_response = self.llm_generator.chat(messages=conversations,
                                    sampling_params=self.sampling_params,
                                    use_tqdm=False)
            
            text_response = llm_response[0].outputs[0].text
            
            llm_time = time.time() - llm_time
            latency_cost['llm_time'] = llm_time
            ### check repeat or batch generation 
            torch.cuda.empty_cache()
            ### extract motion & speech ids & update conversation
            
            gen_time = time.time()
            
            if method == 'llm+speech':
                cleaned_speech_response = self.clean_speech_text_response(text_response)
                cleaned_speech_response = self.shorten_text_response(cleaned_speech_response)
            else:
                cleaned_speech_response, cleaned_motion_response = self.clean_speech_motion_text_response(text_response)
                cleaned_speech_response = self.shorten_text_response(cleaned_speech_response)
                pass
                
            if method == 'llm+speech':
                if cleaned_speech_response:
                    conv.append_message(chatbot_name_, '<speech>' + cleaned_speech_response +'</speech>')
                    print('output message: ', cleaned_speech_response)
                    # save_path = os.path.join(save_dir, f"pred_{conv.get_message_rounds()-1}.wav")
                    self.xtts_model.tts_to_file(text=cleaned_speech_response, speaker_wav=conv.agent_voice_prompt, language="en", file_path=output_audio_path)
                    np.savez(output_motion_path, **{})
                    gen_time = time.time() - gen_time
                    latency_cost['gen_time'] = gen_time
                    overall_time = time.time() - start_time
                    latency_cost['overall_time'] = overall_time
                    return {
                        'speech_text': cleaned_speech_response,
                        'latency': latency_cost,
                    }
            elif method in ['dlp+retrieval', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']:
                if cleaned_speech_response and cleaned_motion_response:
                    print(f"output message: {cleaned_speech_response}, {cleaned_motion_response}")
                    conv.append_message(chatbot_name_, '<speech>' + cleaned_speech_response +'</speech> <motion>' + cleaned_motion_response + '</motion>')
                    # save_path = os.path.join(save_dir, f"pred_{conv.get_message_rounds()-1}.wav")
                    self.xtts_model.tts_to_file(text=cleaned_speech_response, speaker_wav=conv.agent_voice_prompt, language="en", file_path=output_audio_path)
             
                    if method in ['dlp+retrieval', 'dlp+motiongpt+retrieval']:
                        motion_query_embedding = self.text_embedding_model.encode_queries([cleaned_motion_response])
                        re_actions, re_motions = self.unified_dataset.retrieval_motion_by_embeddings(motion_query_embedding[0], top_k=5)
                        gen_motion = None
                        for motion_id in re_motions:
                            gen_motion = self.raw_motion_dataset.get_data_from_name(motion_id)[1]
                            if gen_motion is not None:
                                break
                    else:
                        gen_motion = self.generate_motion(cleaned_motion_response)
                    
                    gen_time = time.time() - gen_time
                    latency_cost['gen_time'] = gen_time
                    
                    if gen_motion is not None:
                        
                        trans = gen_motion[..., :3]
                        poses = gen_motion[..., 3:].reshape(-1, 52, 3)
                        poses = np.concatenate([poses[..., :22, :], np.zeros((poses.shape[0], 3, 3)), poses[..., 22:, :]], axis=1)
                        poses = poses.reshape(-1, 165)
                        np.savez(output_motion_path, trans=trans, poses=poses)
                        
                        overall_time = time.time() - start_time
                        latency_cost['overall_time'] = overall_time
                        
                        return {
                            'speech_text': cleaned_speech_response,
                            'latency': latency_cost,
                            'motion_params': gen_motion,
                        }
                    pass
    
        return None


    def conv_inference_server_local(self, input_motion_path, input_audio_path, output_motion_path, output_audio_path, session_id):
        res = asyncio.run(self._conv_inference_server(input_motion_path, input_audio_path, output_motion_path, output_audio_path, session_id))
        return res
    
    
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
            return self.session_id_to_conversations[session_id].character
        else:
            print(f"Session {session_id} does not exist.")
            return None

    

if __name__ == "__main__":
    # initialize_debugpy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--method", type=str, default='llm+speech')
    parser.add_argument("--model_path", type=str, 
                        default="/root/pengyang/codebase/SOLAMI/extra/meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output_dir", type=str,
                        default="/root/pengyang/codebase/SOLAMI/models/vla/infer_output/llama2_dlp_retrieval_inference-test")
    args = parser.parse_args()


    speech_dir = "SOLAMI_data/IT_Speech/raw_data"

    model_path = args.model_path
    tokenizer_path = os.path.join(model_path, "tokenizer.model")
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.method not in ['llm+speech', 'dlp+retrieval', 'dlp+motiongpt', 'dlp+motiongpt+retrieval']:
        print('Invalid method ', args.method)
        exit()
    conv_model = DLP(model_path)


    test_dataset = ITDataset("SOLAMI_data/Conversation/test_it_items.jsonl",
                             raw_motion_dataset=conv_model.raw_motion_dataset,
                             method=args.method)


    input_motion_path = "/root/pengyang/codebase/SOLAMI/models/vla/data/test_server/motion_user.npz"
    input_audio_path = "SOLAMI_data/audio/voice_prompt/voices/trump_0.mp3"
    
    output_motion_path = "/root/pengyang/codebase/SOLAMI/models/vla/data/test_server/output_character1_0.npz"
    output_audio_path = "/root/pengyang/codebase/SOLAMI/models/vla/data/test_server/output_character1_0.mp3"

    for i in range(3):
        session_id = conv_model.create_session('Trump', 'Trump_0', 'llm+speech')
        res = conv_model.conv_inference_server(input_motion_path, input_audio_path, output_motion_path=output_motion_path, output_audio_path=output_audio_path, session_id='Trump_0')
        conv_model.conv_delete_session('Trump_0')
        print(res['latency'])
    pass
    # for i in tqdm(range(args.part, len(test_dataset), args.period)):
    #     data_item = test_dataset[i]
    #     data_idx = data_item['id']
        

    #     save_dir = os.path.join(output_dir, data_idx)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     else:
    #         continue
        
    #     save_res = {}
        
    #     raw_chat_data = data_item['chat']
    #     character = raw_chat_data[1]['role']
    #     session_id = conv_model.create_session(character)
    #     if session_id is None:
    #         print(f"Session {session_id} is not created at {data_idx}.")
    #         continue
    #     for j, chat_round in enumerate(raw_chat_data):
    #         if chat_round['role'] != character:
    #             motion_id = chat_round['motion_id']
    #             smplx_params = conv_model.raw_motion_dataset.get_data_from_name(motion_id)[1]
                
    #             speech_file_name = chat_round['speech_file_name']
    #             input_speech_path = os.path.join(speech_dir, speech_file_name)
                
    #             response_res = conv_model.conv_inference(smplx_params, input_speech_path, session_id, save_dir)
    #             if response_res is None:
    #                 print(f"Session {session_id} round {j+1} is not valid at {data_idx}.")
    #                 break
    #             else:
    #                 save_res[str(j+1)] = {}
    #                 for key in ['speech_text', 'latency']:
    #                     save_res[str(j+1)][key] = response_res[key]
                    
    #                 if 'motion_params' in response_res:
    #                     if response_res['motion_params'] is not None:
    #                         save_path = os.path.join(save_dir, f"pred_{j+1}.npz")
    #                         np.savez(save_path, **{'motion_params': response_res['motion_params']})
    #         else:
    #             continue
    #     conv_model.conv_delete_session(session_id)
    #     ### save res
    #     with open(os.path.join(save_dir, 'speech_text.json'), 'w', encoding='utf-8') as f:
    #         json.dump(save_res, f, indent=4, ensure_ascii=False)
    #     torch.cuda.empty_cache()
    #     pass
    # print("All Done!")