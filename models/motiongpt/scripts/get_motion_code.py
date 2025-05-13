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
from mGPT.utils.load_checkpoint import load_pretrained_vae

def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

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
    output_dir = os.path.join("SOLAMI_data/tmp_data/tmp_tokens", motion_type + '_' + motion_part)
    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = build_model(cfg, datasets)
    # if hasattr(model, "motion_vae"):
    #     model.vae = model.motion_vae
    # print("model loaded")

    # Strict load vae model
    assert cfg.TRAIN.PRETRAINED_VAE is not None
    load_pretrained_vae(cfg, model)

    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    for batch in tqdm(datasets.train_dataloader(),
                      desc=f'motion tokenize'):
        name = batch['text']
        
        pose = batch['motion']
        pose = pose.cuda().float()

        if pose.shape[1] == 0:
            continue
        if model.vae_hand != None:
            code_pred_body, _ = model.vae_body.encode(pose[..., :model.vae_body.nfeats])
            code_pred_hand, _ = model.vae_hand.encode(pose[..., model.vae_body.nfeats:])
        else:
            code_pred_body, _ = model.vae_body.encode(pose)
            
        motion_tokens = {'body': code_pred_body.cpu().numpy()}
        if model.vae_hand != None:
            motion_tokens['hand'] = code_pred_hand.cpu().numpy()
        
        # target, _ = model.vae.encode(pose)
        # target = target.to('cpu').numpy()

        target_path = os.path.join(output_dir, name[0] + '.npz')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(target_path, **motion_tokens)

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()