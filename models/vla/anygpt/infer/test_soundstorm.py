import sys
sys.path.append("./")
sys.path.append("./anygpt")
import torch
import torchaudio
from speechtokenizer import SpeechTokenizer
from voice_clone import load_soundstorm, semantic2acoustic
from m_utils.prompter import *
from m_utils.anything2token import *
from einops import rearrange
import re
import json


speech_tokenizer_config = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/config.json"
speech_tokenizer_path = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/ckpt.dev"
soundstorm_path = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/soundstorm/speechtokenizer_soundstorm_mls.pt"
DEVICE="cuda"

speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)        
speech_tokenizer.eval()
speech_tokenizer.to(device=DEVICE)
soundstorm = load_soundstorm(soundstorm_path)
soundstorm.eval()
soundstorm.to(device=DEVICE)


def encode_speech(
        audio_path
    ):
        wav, sr = torchaudio.load(audio_path)
        num_frames = wav.size(1)
        duration = num_frames / sr
        print(f"{duration:.2f} seconds")
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1, ]
        if sr != speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).to(DEVICE)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = speech_tokenizer.encode(wav) # codes: (n_q, B, T)
        print(f"codes.shape: {codes.shape}")
        print(f"per second codes.shape: {codes.shape[2] / duration}")
        return codes[0, 0, :]
    
def decode_speech(content, prompt_path=None):
    if prompt_path:
        # get tokens of prompt
        prompt_wav, sr = torchaudio.load(prompt_path)
        prompt_wav = prompt_wav.to(DEVICE)
        if sr != speech_tokenizer.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, speech_tokenizer.sample_rate)
        # If it is stereo, take the average to mono
        if prompt_wav.shape[0] == 2:
            prompt_wav = prompt_wav.mean(dim=0).unsqueeze(0)
        prompt_tokens = rearrange(speech_tokenizer.encode(prompt_wav.unsqueeze(0)), 'q b n -> b n q')
        prompt_tokens = prompt_tokens[:, :, :150]  # only use the 150 frame
    else:
        prompt_tokens = None
    # print(prompt_tokens)
    # codes.shape：(1, 1, n)
    semantic_codes = [[int(num) for num in content]]
    # wav: (b, 1, t)
    config_dict = json.load(open('config/generate_config.json', 'r'))
    wav = semantic2acoustic(torch.Tensor(semantic_codes).int().to(DEVICE), prompt_tokens, 
                            soundstorm, speech_tokenizer, steps=config_dict['vc_steps'])
    wav = wav.squeeze(0).detach().cpu()
    return wav

data_1_path = "audio_dataset/test_data/trump_1.mp3"
data_2_path = "audio_dataset/test_data/trump_2.mp3"

data_1_codes = encode_speech(data_1_path)
data_2_codes = encode_speech(data_2_path)


data_1_decode_only = decode_speech(data_1_codes)
torchaudio.save("/root/pengyang/codebase/SOLAMI/extra/AnyGPT/infer_output/base/data_1_decode_only.wav", 
                data_1_decode_only, 
                speech_tokenizer.sample_rate)

data_1_decode_by_data_2 = decode_speech(data_1_codes, data_2_path)
torchaudio.save("/root/pengyang/codebase/SOLAMI/extra/AnyGPT/infer_output/base/data_1_decode_by_data_2.wav",
                data_1_decode_by_data_2, 
                speech_tokenizer.sample_rate)

data_2_decode_only = decode_speech(data_2_codes)
torchaudio.save("/root/pengyang/codebase/SOLAMI/extra/AnyGPT/infer_output/base/data_2_decode_only.wav",
                data_2_decode_only, 
                speech_tokenizer.sample_rate)

data_2_decode_by_data_1 = decode_speech(data_2_codes, data_1_path)
torchaudio.save("/root/pengyang/codebase/SOLAMI/extra/AnyGPT/infer_output/base/data_2_decode_by_data_1.wav",
                data_2_decode_by_data_1, 
                speech_tokenizer.sample_rate)
