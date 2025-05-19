import os 
import json
import numpy as np
from tqdm import tqdm


for part in ['train', 'test']:
    data_buffer = []
    line_counter = 0

    target_path = os.path.join("/root/pengyang/codebase/SOLAMI/SOLAMI_data/tmp_data/G1ML3D_tokens", 'motion_{}.jsonl'.format(part))

    with open(f"/root/pengyang/codebase/SOLAMI/SOLAMI_data/HumanML3D/{part}.txt", "r") as file:
        lines = file.readlines()
    for line in tqdm(lines):
        try:
            motion = np.load(f"/root/pengyang/codebase/SOLAMI/SOLAMI_data/HumanML3D/TOKENS/{line[:-1]+'.npy'}")[0].tolist()
            with open(f"/root/pengyang/codebase/SOLAMI/SOLAMI_data/HumanML3D/texts/{line[:-1]+'.txt'}", "r") as f_txt:
                text = list(f_txt.readlines())
            line_counter += 1
            motion_id = line.strip()
            data_item = {
                'id': motion_id,
                'motion': motion,
                'text': text,
            }
            data_buffer.append(data_item)
        except Exception as e:
            print(f"Error processing {line[:-1]+'.npy'}", e)
            continue

        if len(data_buffer) >= 500:
            with open(target_path, 'a', encoding='utf-8') as f:
                for item in data_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print('Processed {} lines'.format(line_counter))
            data_buffer = []
    # target_path = os.path.join(output_dir, name[0] + '.npz')
    # Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    # np.savez(target_path, **motion_tokens)
    if data_buffer:
        with open(target_path, 'a', encoding='utf-8') as f_out:
            for item in data_buffer:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")