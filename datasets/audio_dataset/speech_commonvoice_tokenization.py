import sys
import os
sys.path.append("/root/pengyang/codebase/SOLAMI/models/vla/")
sys.path.append("/root/pengyang/codebase/SOLAMI/models/vla/anygpt")
import torch
import torchaudio
from speechtokenizer import SpeechTokenizer
from m_utils.prompter import *
from m_utils.anything2token import *
from einops import rearrange
import re
import json
from m_utils.loggings import get_logger
import fire
import torch.distributed as dist
import debugpy
from tqdm import tqdm
import pandas as pd




def process_speech_data(part=0, period=4, gpu_id=0, maxlen=150000):
    DEVICE = 'cuda:%d'%gpu_id 
    speech_tokenizer_config = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/config.json"
    speech_tokenizer_path = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/ckpt.dev"

    speech_meta_path = "SOLAMI_data/audio/commonvoice_en/validated.tsv"
    speech_dir = "SOLAMI_data/audio/commonvoice_en/clips"
    output_dir = "SOLAMI_data/audio/commonvoice_processed"
    output_path = os.path.join(output_dir, "commonvoice_{}_{}_{}.jsonl".format(part, period, maxlen))

    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)        
    speech_tokenizer.eval()
    speech_tokenizer.to(device=DEVICE)


    def encode_speech(
            audio_path,
            logger=None,
        ):
        wav, sr = torchaudio.load(audio_path)
        num_frames = wav.size(1)
        duration = num_frames / sr
        if logger is not None:
            audio_path_log = os.path.join(*audio_path.split("/")[-2:])
            # logger.info(f"Path: {audio_path_log} duration: {duration:.2f} seconds")
        # monophonic checking
        if wav.shape[0] > 1:
            wav = wav[:1, ]
        if sr != speech_tokenizer.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, speech_tokenizer.sample_rate)
        wav = wav.unsqueeze(0).to(DEVICE)
        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = speech_tokenizer.encode(wav) # codes: (n_q, B, T)
        return codes[0, 0, :]


    logger = get_logger(local_rank=0, save_path=os.path.join(output_dir, "speech_process.log"), log_level='info')

    line_counter = 0
    data_buffer = []
    
    # with open(speech_meta_path, 'r', encoding='utf-8') as f:
    #     total_lines = sum(1 for _ in f)
    
    df = pd.read_csv(speech_meta_path, sep='\t')
    num_rows = df.shape[0]
    processed_lines = min(num_rows, maxlen)
    print(f"processed lines: {processed_lines}, Total lines: {num_rows}")
    
    line_counter = 0
    data_buffer = []
    
    for index, line in tqdm(df.iterrows(), total=processed_lines, desc="Processing CommonVoice", unit="line"):
        line_counter += 1
        if line_counter % period != part:
            continue
        
        if line_counter > processed_lines:
            break
        
        processed_data = {}
        processed_data['id'] = f"commonvoice_{line_counter}"
        processed_data['chat'] = []
        
        text = line['sentence']
        role = 'user'
        try:
            speech_path = os.path.join(speech_dir, line['path'])
            speech_code = encode_speech(speech_path, logger)
        except:
            logger.error(f"Error processing {speech_path}")
            continue
        speech_str = modality_tokens_to_string(speech_code, modality="speech")
        data_item = {
            'role': role,
            "speech_path": speech_path,
            "text": text,
            "speech": speech_str,
        }
        processed_data['chat'].append(data_item)
        data_buffer.append(processed_data)
        
        if len(data_buffer) >= 50:
            with open(output_path, 'a', encoding='utf-8') as f:
                for item in data_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info('Processed {} lines in the {}-th part'.format(line_counter, part))
            data_buffer = []
                
    if data_buffer:
        with open(output_path, 'a', encoding='utf-8') as f_out:
            for item in data_buffer:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        # chat_json = json.loads(line.strip())
        

if __name__ == "__main__":
    fire.Fire(process_speech_data)