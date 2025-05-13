import sys
import os
sys.path.append("/root/pengyang/codebase/SOLAMI/models/vla/")
sys.path.append("/root/pengyang/codebase/SOLAMI/models/vla/anygpt")
import json
from tqdm import tqdm
import torch
import torchaudio
from speechtokenizer import SpeechTokenizer
from m_utils.prompter import *
from m_utils.anything2token import *
from m_utils.loggings import get_logger
## get conversation items
import torch.distributed as dist
import debugpy


def initialize_debugpy():
    print("Debugpy is listening on port 15696")
    debugpy.listen(("0.0.0.0", 15696))
    debugpy.wait_for_client()

# initialize_debugpy()



def get_agent_name_from_descriptions(des):
    if 'Trump' in des:
        return 'Trump'
    elif '11-45-G' in des:
        return '11-45-G'
    elif 'Bananya' in des:
        return 'Banaya'
    elif 'Batman' in des:
        return 'Batman'
    elif 'Link' in des:
        return 'Link'
    elif 'Samantha' in des:
        return 'Samantha'
    else:
        return 'User'


## check whether the item is right
def check_chat_item(chat):
    if len(chat) > 2:
        return False
    # check the values of chat 0 is None or have None in the list
    for key in ['body', 'hand', 'trans']:
        if chat[0][key] is None or None in chat[0][key]:
            return False
    return True

#### avaiable data items


save_dir = "SOLAMI_data/Conversation"
speech_dir = "SOLAMI_data/IT_Speech/raw_data"
valid_speech_files = os.listdir(speech_dir)

motion_file_paths = [
    'SOLAMI_data/tmp_data/pretrain_new_tokens/local_cont6d_body_hand_sep/motion_train_merged.jsonl',
    'SOLAMI_data/tmp_data/pretrain_new_tokens/local_cont6d_body_hand_sep/motion_test_merged.jsonl',
]

id_to_motion_items = {}
for motion_file_path in motion_file_paths:
    with open(motion_file_path, 'r') as f:
        for line in f:
            data_item = json.loads(line)
            chat = data_item['chat']
            for chat_item in chat:
                if chat_item['motion_id'] in id_to_motion_items:
                    # print(f"Duplicate motion id {chat_item['motion_id']}")
                    pass
                else:
                    id_to_motion_items[chat_item['motion_id']] = chat_item
print(f"Total motion items: {len(id_to_motion_items)}")


train_save_file = os.path.join(save_dir, "train_original_data_items.jsonl")
test_save_file = os.path.join(save_dir, "test_original_data_items.jsonl")
train_data_items = []
test_data_items = []
if os.path.exists(train_save_file) and os.path.exists(test_save_file):
    with open(train_save_file, 'r') as f:
        for line in f:
            train_data_items.append(json.loads(line))
    print(f"Load train data items from {train_save_file}, total: {len(train_data_items)}")
    with open(test_save_file, 'r') as f:
        for line in f:
            test_data_items.append(json.loads(line))
    print(f"Load test data items from {test_save_file}, total: {len(test_data_items)}")

else:
    print(f"Train or test data items not found, start to generate")

    source_data_dir = "/root/pengyang/codebase/SOLAMI/multimodal_gen/data_gen/output/sim_all"
    json_file_list = os.listdir(source_data_dir)

    valid_data_items = []
    for json_file in tqdm(json_file_list):
        with open(os.path.join(source_data_dir, json_file), 'r') as f:
            data_list = json.load(f)
            data = data_list[0]
            role_to_speaker_id = {}
            role_to_speaker_id['A'] = 'User'
            role_to_speaker_id['B'] = get_agent_name_from_descriptions(data['agent_settings'])
            data_id = data['data_id']
            dialogs = data['dialogs']
            valid_flag = True
            for round_id, behavior in dialogs.items():
                #### check speech, check motion
                agent_name = role_to_speaker_id[behavior['role']]
                speech_file_name = json_file.split('.')[0] + '___' + f"0___{round_id}___{agent_name}.wav"
                if speech_file_name not in valid_speech_files:
                    valid_flag = False
                motion_id = behavior['action_index']
                if motion_id not in id_to_motion_items:
                    valid_flag = False
            if valid_flag:
                valid_data_items.append(data)

    print(f"Total valid data items: {len(valid_data_items)}")

    ## split valid data items into train and test  9:1, every 10 items, 9 for train, 1 for test

    for i, data_item in enumerate(valid_data_items):
        if i % 10 == 0:
            test_data_items.append(data_item)
        else:
            train_data_items.append(data_item)

    ### save valid data items as jsonl

    with open(train_save_file, 'w') as f:
        for data_item in train_data_items:
            f.write(json.dumps(data_item) + '\n')
    print(f"Save train data items to {train_save_file}")

    with open(test_save_file, 'w') as f:
        for data_item in test_data_items:
            f.write(json.dumps(data_item) + '\n')
    print(f"Save test data items to {test_save_file}")


#### tokenize the data items
speech_tokenizer_config = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/config.json"
speech_tokenizer_path = "/root/pengyang/codebase/SOLAMI/extra/AnyGPT-speech-modules/speechtokenizer/ckpt.dev"

DEVICE = f"cuda:0"

def encode_speech(
        audio_path,
        logger=None,
    ):
    wav, sr = torchaudio.load(audio_path)
    num_frames = wav.size(1)
    duration = num_frames / sr
    if logger is not None:
        audio_path_log = os.path.join(*audio_path.split("/")[-2:])
        # logger.info(f"Path: {audio_path_log} duration: {duration:.2f} 秒")
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

speech_tokenizer = SpeechTokenizer.load_from_checkpoint(speech_tokenizer_config, speech_tokenizer_path)        
speech_tokenizer.eval()
speech_tokenizer.to(device=DEVICE)

logger = get_logger(local_rank=0, save_path=os.path.join(save_dir, "speech_process.log"), log_level='info')

data_valid_dicts = {
    'train': train_data_items,
    'test': test_data_items,
}

for key, data_items in data_valid_dicts.items():
    data_buffer = []
    save_path = os.path.join(save_dir, f"{key}_it_items.jsonl")
    for data_item in tqdm(data_items):
        dialogs = data_item['dialogs']
        role_to_speaker_id = {}
        role_to_speaker_id['A'] = 'User'
        role_to_speaker_id['B'] = get_agent_name_from_descriptions(data_item['agent_settings'])
        processed_data = {}
        processed_data['id'] = data_item['data_id']
        processed_data['topic'] = data_item['topic']
        processed_data['chat'] = []
        for round_id, behavior in dialogs.items():
            agent_name = role_to_speaker_id[behavior['role']]
            speech_file_name = 'sim_all__' + data_item['data_id'] + '___' + f"0___{round_id}___{agent_name}.wav"
            speech_file_path = os.path.join(speech_dir, speech_file_name)
            speech_code = encode_speech(speech_file_path, logger=logger)
            motion_id = behavior['action_index']
            motion_item = id_to_motion_items[motion_id]
            processed_data['chat'].append({
                'speech': speech_code.cpu().numpy().tolist(),
                'speech_text': behavior['speech'],
                'speech_file_name': speech_file_name,
                'role': agent_name,
                'motion_text': motion_item['text'],
                'body': motion_item['body'],
                'hand': motion_item['hand'],
                'trans': motion_item['trans'],
                'motion_id': motion_id,
                'motion_des': behavior['motion'],
            })
            pass
        data_buffer.append(processed_data)
        if len(data_buffer) >= 500:
            with open(save_path, 'a', encoding='utf-8') as f:
                for item in data_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Processed {len(data_buffer)} lines")
            data_buffer = []
    if len(data_buffer) > 0:
        with open(save_path, 'a', encoding='utf-8') as f:
            for item in data_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Processed {len(data_buffer)} lines")