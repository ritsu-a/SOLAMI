import os
import sys
sys.path.append("/root/pengyang/codebase/SOLAMI/models/motiongpt")
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla')
from omegaconf import OmegaConf

from mGPT.config import parse_args, instantiate_from_config
from mGPT.config import instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.utils.load_checkpoint import load_pretrained_vae, load_pretrained
import torch
from motion.smplx_process import process_smplx_feature

import torch.distributed as dist
import debugpy

def initialize_debugpy():
    # if not dist.is_initialized() or dist.get_rank() == 0:
        # print(f"Rank: {dist.get_rank()} - Debugpy is listening on port 15696")
        print("Debugpy is listening on port 15696")
        debugpy.listen(("0.0.0.0", 15696))
        debugpy.wait_for_client()


def main():
    cfg = OmegaConf.load("/root/pengyang/codebase/SOLAMI/models/motiongpt/experiments/mgpt/Pretrain_HumanML3D_GPT2_Local_Body_Hand_Sep_NoInterleave/config_2024-09-24-20-26-38_train.yaml")
    cfg.TEST.CHECKPOINTS = "/root/pengyang/codebase/SOLAMI/models/motiongpt/experiments/mgpt/Pretrain_HumanML3D_GPT2_Local_Body_Hand_Sep_NoInterleave/checkpoints/last.ckpt"
    cfg.TEST.BATCH_SIZE = 1
    cfg.EXPER.transform = False
    cfg.DEBUG = True
    
    cfg.METRIC.TM2T.t2m_path = "/root/pengyang/codebase/SOLAMI/models/motiongpt/deps/t2m"
    cfg.METRIC.TYPE = []
    cfg.lm.gpt2_medium.params.model_path = "/root/pengyang/codebase/SOLAMI/models/motiongpt/deps/gpt2"
    datasets = build_data(cfg, phase='token')
    
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['params']['cfg'] = cfg
    model_config['params']['datamodule'] = datasets
    model = instantiate_from_config(model_config)
    
    load_pretrained(cfg, model, phase="test", strict=False)
    model = model.to('cuda')
    model.eval()
    input_text = "wave hands"
    repeat_samples = 3
    outputs = model.lm.generate_conditional([input_text] * repeat_samples, task="t2m")
    for i in range(len(outputs)):
        outputs[i] = torch.clamp(outputs[i], 0, 512 - 1, out=None)
        assert len(outputs[i]) == 2
        if len(outputs[i][0]) > 1:
            body_feat = model.vae_body.decode(outputs[i][0])
            hand_feat = model.vae_hand.decode(outputs[i][1])
            motion = torch.cat((body_feat, hand_feat), dim=-1)
            break
        else:
            motion = None
        # if len(outputs[i]) > 1:
        #     motion = self.vae.decode(outputs[i])
        # else:
        #     motion = torch.zeros_like(feats_ref[i:i + 1, ...])

    if motion is not None:
        motion_params = model.feats2joints(motion)
    else:
        motion_params = None
        
    input_motion = motion_params
    motion_feature = process_smplx_feature(input_motion)
    mean = torch.tensor(model.datamodule.hparams.mean, dtype=torch.float32).to('cuda')
    std = torch.tensor(model.datamodule.hparams.std, dtype=torch.float32).to('cuda')
    motion_feature_input = (motion_feature - mean) / std
    
    code_pred_body, _ = model.vae_body.encode(motion_feature_input[..., :model.vae_body.nfeats])
    code_pred_hand, _ = model.vae_hand.encode(motion_feature_input[..., model.vae_body.nfeats:])
    code_pred = torch.cat([code_pred_body, code_pred_hand], dim=0)
    
    motion_tokens = [code_pred] * repeat_samples
    lengths_tokens = [len(code_pred[0])] * repeat_samples
    outputs = model.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths_tokens,
                                               task="m2t",
                                               stage='test')
    
    print("Model loaded")

if __name__ == "__main__":
    # initialize_distributed()
    initialize_debugpy()
    main()