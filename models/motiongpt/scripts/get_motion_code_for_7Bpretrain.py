import os
print(os.getcwd())
import sys
sys.path.append("/root/pengyang/codebase/SOLAMI/models/motiongpt")
os.environ["TRANSFORMERS_CACHE"] = "~/.cache/huggingface/hub"
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained_vae, load_pretrained
import json


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 1

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")
    output_dir = os.path.join(datasets.hparams.data_root, cfg.DATASET.CODE_PATH)
    motion_type = cfg['EXPER']['motion_repre']
    motion_part = cfg['EXPER']['motion_part']
    output_dir = os.path.join("SOLAMI_data/tmp_data/pretrain_new_tokens", motion_type + '_' + motion_part)
    output_dir = output_dir.replace(' ', '_')
    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = build_model(cfg, datasets)
    # if hasattr(model, "motion_vae"):
    #     model.vae = model.motion_vae
    # print("model loaded")

    # Strict load vae model
    
    if cfg.TRAIN.PRETRAINED:
        load_pretrained(cfg, model)
    
    assert cfg.TRAIN.PRETRAINED_VAE is not None
    load_pretrained_vae(cfg, model)

    save_state_dir = "/root/pengyang/codebase/SOLAMI/extra/motion_tokenizer_final"
    torch.save(model.vae_body.state_dict(), os.path.join(save_state_dir, 'body.pth'))
    torch.save(model.vae_hand.state_dict(), os.path.join(save_state_dir, 'hand.pth'))
    torch.save(model.vae_transform.state_dict(), os.path.join(save_state_dir, 'transform.pth'))

    model.eval()
    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    data_generators = {
        'train': datasets.train_dataloader(),
        'test': datasets.test_dataloader(),
    }    
    for part in ['train', 'test']:
        print(f'Processing {part} data')
        print(f'Length {len(data_generators[part])}')
        data_generator = data_generators[part]
        data_buffer = []
        line_counter = 0
        target_path = os.path.join(output_dir, 'motion_{}.jsonl'.format(part))
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        for batch in tqdm(data_generator, desc=f'motion tokenize'):
            line_counter += 1
            motion_id = batch['text'][0]
            
            pose = batch['motion']
            pose = pose.cuda().float()
            # print(pose.shape[1])
            if pose.shape[1] == 0:
                continue
            try:
                if model.vae_transform != None:
                    code_pred_transform, _ = model.vae_transform.encode(batch['transform'].cuda().float())
                else:
                    code_pred_transform = None
                
                if model.vae_hand != None:
                    code_pred_body, _ = model.vae_body.encode(pose[..., :model.vae_body.nfeats])
                    code_pred_hand, _ = model.vae_hand.encode(pose[..., model.vae_body.nfeats:])
                else:
                    code_pred_body, _ = model.vae_body.encode(pose)
            except Exception as e:
                print(e)
                continue
            
            data_item = {
                'id': motion_id,
                'chat': []
            }
            
            text = batch['all_captions'][0][:3]
            body_tokens = code_pred_body.cpu().numpy().tolist()[0]
            hand_tokens = code_pred_hand.cpu().numpy().tolist()[0]
            trans_tokens = code_pred_transform.cpu().numpy().tolist()[0]
            partner_id = batch['all_captions'][0][3]
            data_item['chat'].append({
                'text': text,
                'body': body_tokens,
                'hand': hand_tokens,
                'trans': trans_tokens,
                'motion_id': motion_id,})
            if partner_id != None:
                data_item['chat'].append({
                    'motion_id': partner_id,})
            data_buffer.append(data_item)
            # motion_tokens = {'body': code_pred_body.cpu().numpy()}
            # if model.vae_hand != None:
            #     motion_tokens['hand'] = code_pred_hand.cpu().numpy()
            
            # target, _ = model.vae.encode(pose)
            # target = target.to('cpu').numpy()
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

    print(
        f'Motion tokenization done, the motion tokens are saved to {target_path}'
    )


if __name__ == "__main__":
    main()