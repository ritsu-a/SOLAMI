import os
import sys
os.environ['HOME'] = 'YOUR_HOME_PATH'
sys.path.append('SOLAMI')
project_paths = [
    '/root/pengyang/codebase/SOLAMI/extra/ChatTTS',
    '/root/pengyang/codebase/SOLAMI/extra/OpenVoice',
    '/root/pengyang/codebase/SOLAMI/extra/MeloTTS',
]
for project_path in project_paths:
    project_path = os.path.realpath(project_path)
    if project_path not in sys.path:
        sys.path.append(project_path)
        
import time
import logging     
import torch
import json
import warnings
import torchaudio
import ChatTTS
from openai import OpenAI, AzureOpenAI
from melo.api import TTS as OpenVoice_TTS
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODELS = {}

def load_TTS_model(model_name: str='openvoice-tts', device: str="cuda:0"):
    load_model_time = 0
    if model_name == 'openvoice-tts':
        if model_name not in MODELS:
            start_time = time.time()
            tts_model = OpenVoice_TTS(language='EN', device=device)
            MODELS[model_name] = tts_model
            end_time = time.time()
            load_model_time = end_time - start_time
            logging.info(f"Model {model_name} loaded for {load_model_time} seconds.")
    elif model_name == 'chat-tts':
        if model_name not in MODELS:
            start_time = time.time()
            chat = ChatTTS.Chat()
            chat.load(source='huggingface', compile=True)
            MODELS[model_name] = chat
            load_model_time = end_time - start_time
            logging.info(f"Model {model_name} loaded for {load_model_time} seconds.")
        pass
    elif model_name == 'oai-tts':
        pass
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
def TTS_Infer(model_name: str='openvoice-tts', voice: str ='EN-Default', input_text: str='', output_path: str=''):
    if model_name == 'openvoice-tts':
        tts_model = MODELS[model_name]
        texts = input_text
        start_time = time.time()
        speaker_ids = tts_model.hps.data.spk2id
        if voice in speaker_ids.keys():
            speaker_id = speaker_ids[voice]
            tts_model.tts_to_file(texts, speaker_id, output_path, speed=1.0)
        end_time = time.time()
        return end_time - start_time
    if model_name == 'chat-tts':
        tts_model = MODELS[model_name]
        texts = [input_text,]
        start_time = time.time()
        wavs = tts_model.infer(texts)
        end_time = time.time()
        torchaudio.save(output_path, torch.from_numpy(wavs[0]).to(torch.float32), 24000)
        return end_time - start_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1/sim1__1200_0_if_sc.json')
    args = parser.parse_args()
    
    json_path = args.json_path
    with open(json_path, 'r') as f:
        data = json.load(f)

    data = data[0]
    tone_mappings = {
        'normal person': 'EN-Default',
        '3D virtual companion': 'EN-BR',    
        'Donald Trump': 'EN-US',
        "Batmam": 'EN-US',
        "Link": "EN-US",
        "Bananya": "EN-US",
        "11-45-G": "EN-US",
    }

    output_dir = "/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/output/sim1_audio"
    basename = os.path.basename(json_path).split('.')[0]
    output_dir = os.path.join(output_dir, basename)
    os.makedirs(output_dir, exist_ok=True)
    load_TTS_model(model_name='openvoice-tts', device=DEVICE)


    tone = {"A": '', "B": ''}
    for key, value in tone_mappings.items():
        if key in data['user_settings']:
            tone["A"] = value
        if key in data['agent_settings']:
            tone["B"] = value

    for index, (round, dialog) in enumerate(data['dialogs'].items()):
        tone_tmp = tone[dialog['role']]
        infer_time = TTS_Infer(model_name='openvoice-tts', voice=tone_tmp, input_text=dialog['speech'], output_path=os.path.join(output_dir, f'{round}.wav'))

    logging.info(f"Audio files saved to {output_dir}")
    # {'EN-US': 0, 'EN-BR': 1, 'EN_INDIA': 2, 'EN-AU': 3, 'EN-Default': 4}