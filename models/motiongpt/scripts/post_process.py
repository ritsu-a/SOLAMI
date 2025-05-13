import os
print(os.getcwd())
import sys
sys.path.append("/root/pengyang/codebase/SOLAMI/models/motiongpt")
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
import json
import copy



## check whether the item is right
def check_chat_item(chat):
    if len(chat) > 2:
        return False
    # check the values of chat 0 is None or have None in the list
    for key in ['body', 'hand', 'trans']:
        if chat[0][key] is None or None in chat[0][key]:
            return False
    return True
    

def main():
    
    data_file_paths = [
        'SOLAMI_data/tmp_data/pretrain_new_tokens/local_cont6d_body_hand_sep/motion_train.jsonl',
        'SOLAMI_data/tmp_data/pretrain_new_tokens/local_cont6d_body_hand_sep/motion_test.jsonl',
    ]
    
    for data_file_path in data_file_paths:
        data_items = []
        with open(data_file_path, 'r') as file:
            for line in file:
                item = json.loads(line)
                if check_chat_item(item['chat']):
                    data_items.append(item)
                else:
                    print(f"Invalid item: {item}")

        id_to_chat_item = {}
        
        for data_item in data_items:
            id_ = data_item['id']
            chat = copy.deepcopy(data_item['chat'][0])
            if id_ in id_to_chat_item:
                pass
                # print(f"Duplicate id {id_}")
            else:
                id_to_chat_item[id_] = chat
            
        for data_item in data_items:
            if len(data_item['chat']) == 1:
                continue
            if data_item['chat'][1]['motion_id'] in id_to_chat_item:
                item = id_to_chat_item[data_item['chat'][1]['motion_id']]
                if item['hand'] is None:
                    continue
                data_item['chat'][1] = item
            else:
                # bug here!
                data_item['chat'] = data_item['chat'][:1]
        
        new_data_path = data_file_path.replace('.jsonl', '_merged.jsonl')
        with open(new_data_path, 'w') as file:
            for data_item in data_items:
                file.write(json.dumps(data_item) + '\n')
        print(f"Saved to {new_data_path}")
        #     data_item = {
        #         'id': motion_id,
        #         'chat': []
        #     }
            
        #     text = batch['all_captions'][0][:3]
        #     body_tokens = code_pred_body.cpu().numpy().tolist()[0]
        #     hand_tokens = code_pred_hand.cpu().numpy().tolist()[0]
        #     trans_tokens = code_pred_transform.cpu().numpy().tolist()[0]
        #     partner_id = batch['all_captions'][0][3]
        #     data_item['chat'].append({
        #         'text': text,
        #         'body': body_tokens,
        #         'hand': hand_tokens,
        #         'trans': trans_tokens,
        #         'motion_id': motion_id,})
        #     if partner_id != None:
        #         data_item['chat'].append({
        #             'motion_id': partner_id,})
        #     data_buffer.append(data_item)
        #     # motion_tokens = {'body': code_pred_body.cpu().numpy()}
        #     # if model.vae_hand != None:
        #     #     motion_tokens['hand'] = code_pred_hand.cpu().numpy()
            
        #     # target, _ = model.vae.encode(pose)
        #     # target = target.to('cpu').numpy()
        #     if len(data_buffer) >= 500:
        #         with open(target_path, 'a', encoding='utf-8') as f:
        #             for item in data_buffer:
        #                 f.write(json.dumps(item, ensure_ascii=False) + '\n')
        #         print('Processed {} lines'.format(line_counter))
        #         data_buffer = []
        #     # target_path = os.path.join(output_dir, name[0] + '.npz')
        #     # Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        #     # np.savez(target_path, **motion_tokens)
        # if data_buffer:
        #     with open(target_path, 'a', encoding='utf-8') as f_out:
        #         for item in data_buffer:
        #             f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    # print(
    #     f'Motion tokenization done, the motion tokens are saved to {target_path}'
    # )


if __name__ == "__main__":
    main()