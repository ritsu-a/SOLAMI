import numpy as np
import os
import random
import torch
import time
from mGPT.config import instantiate_from_config
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
import sys
import mGPT.render.matplot.plot_3d_global as plot_3d
sys.path.append('/mnt/AFS_jiangjianping/projects/tools/smplx')
import smplx

class MotionGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()
        
        self.cfg = cfg
        # Instantiate motion tokenizer
        if motion_vae != None:
            self.vae = instantiate_from_config(motion_vae)
       
        # # TODO
        # self.vae_transform = None
        # Instantiate motion-language model
        self.lm = instantiate_from_config(lm)
  

        # Freeze the motion tokenizer for lm training
        if 'lm' in self.hparams.stage:
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False
        
        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))

    def get_smplx_model(self):
        self.betas = torch.tensor([-0.06134899, -0.4861751 ,  0.8630473 , -3.07320443,  1.10772016,
                                    -1.44656493,  2.97690664, -1.12731489,  1.24817344, -1.4111463 ,
                                    -0.04035034, -0.29547926,  0.38509519,  0.13750311,  0.94445029,
                                    -0.47172116], dtype=torch.float32)
        
        self.t_root_J = torch.tensor([
            0, 0, 0
        ], dtype=torch.float32)
        
        self.model_path = '/mnt/AFS_jiangjianping/datasets/Assets/SMPL_MODELS/smplx/SMPLX_MALE.npz'
        self.smplx_model = smplx.create(self.model_path, 
                                        model_type='smplx', 
                                        gender='male', 
                                        ext='npz', 
                                        num_betas=len(self.betas), 
                                        use_pca=False, 
                                        flat_hand_mean=True)
        self.smplx_model.eval()

    def smplx_infer(self, res):
        batch_size, seq_len, feat_len = res.shape
        
        global_orient = res[..., 3:6]
        
        transl = res[..., :3] - self.t_root_J.to(res.device)
        
        betas = self.betas.to(res.device)
        betas = betas.repeat(batch_size, seq_len, 1)
        
        expression = torch.zeros([batch_size, seq_len, 10], dtype=torch.float32).to(res.device)
        
        if feat_len in [6 + 21*3, 6+24*3]:
            jaw_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            leye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            reye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            left_hand_pose = torch.zeros([batch_size, seq_len, 45], dtype=torch.float32).to(res.device)
            right_hand_pose = torch.zeros([batch_size, seq_len, 45], dtype=torch.float32).to(res.device)
            body_pose = res[..., 6:6 + 21*3]
        elif feat_len == 6 + 51 *3:
            body_pose = res[..., 6:6+21*3]
            left_hand_pose = res[..., 6+21*3: 6+(21+15)*3]
            right_hand_pose = res[..., 6+36*3:]
            jaw_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            leye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
            reye_pose = torch.zeros([batch_size, seq_len, 3], dtype=torch.float32).to(res.device)
        elif feat_len == 6 + 54 *3:
            body_pose = res[..., 6:6+21*3]
            jaw_pose = res[..., 6+21*3: 6+22*3]
            leye_pose = res[..., 6+22*3: 6+23*3]
            reye_pose = res[..., 6+23*3: 6+24*3]
            left_hand_pose = res[..., 6+24*3: 6+(24+15)*3]
            right_hand_pose = res[..., 6+39*3:]
        else:
            raise ValueError('Unknown feature length')
        
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
        return joints[:, :, :55]

    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]

        # Forward
        # texts = ['Generate motion: ' + text for text in texts]
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(texts)):
            if task == "pred":
                motion = self.vae.decode(
                    torch.cat((batch["motion"][i], outputs[i])))
            elif task in ["t2m", "m2t", "inbetween"]:
                motion = self.vae.decode(outputs[i])
                # motion = self.datamodule.denormalize(motion)
                lengths.append(motion.shape[1])
            else:
                raise NotImplementedError

            if motion.shape[1] > max_len:
                max_len = motion.shape[1]

            if task in ["t2m", "m2t", "pred"]:
                feats_rst_lst.append(motion)

            elif task == "inbetween":
                motion = torch.cat(
                    (batch["motion_heading"][i][None],
                     motion[:, lengths_ref[i] // 4:lengths_ref[i] // 4 * 3,
                            ...], batch["motion_tailing"][i][None]),
                    dim=1)
                feats_rst_lst.append(motion)

        feats_rst = torch.zeros(
            (len(feats_rst_lst), max_len, motion.shape[-1])).to(self.device)

        # padding and concat
        for i in range(len(feats_rst_lst)):
            feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]

        # Recover joints for evaluation
        joints_rst = self.feats2joints(feats_rst)

        # return set
        outputs = {
            "texts": output_texts,
            "feats": feats_rst,
            "joints": joints_rst,
            "length": lengths
        }

        return outputs

    def train_lm_forward(self, batch):
        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        all_captions = batch['all_captions']
        partner_motion = batch['partner_motion']
        if self.hparams.condition == 'caption':
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]

        if hasattr(self.hparams.cfg.TRAIN, "TASK") and self.hparams.cfg.TRAIN.TASK == "interaction":
                b_motion = batch["b_motion"]
                b_speech = batch["b_speech"]
                b_length = batch["b_length"]
                outputs = self.lm(texts, tokens_ref, lengths, tasks, b_speech=b_speech, b_motion=b_motion, b_length=b_length)
        else:      
            # LLM Forward
            outputs = self.lm(texts, tokens_ref, lengths, tasks, partner_motion=partner_motion)
        # outputs = self.t2m_gpt.generate(texts)
        return {'outputs': outputs}

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            instructions = pjoin(self.datamodule.hparams.data_root,
                                 'template_instructions.json')
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        min_len = lengths.copy()
        # Forward
        outputs = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks)

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)

        pred_valid = []
        
        for i in range(len(texts)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)
            if self.vae_hand != None:
                assert len(outputs[i]) == 2
                if len(outputs[i][0]) > 1:
                    body_feat = self.vae_body.decode(outputs[i][0])
                    hand_feat = self.vae_hand.decode(outputs[i][1])
                    motion = torch.cat((body_feat, hand_feat), dim=-1)
                    pred_valid.append(True)
                else:
                    motion = torch.zeros_like(feats_ref[i:i + 1, ...])
                    pred_valid.append(False)
            else:
                if len(outputs[i][0]) > 1:
                    motion = self.vae_body.decode(outputs[i])
                    pred_valid.append(True)
                else:
                    motion = torch.zeros_like(feats_ref[i:i + 1, ...])
                    pred_valid.append(False)
            # if len(outputs[i]) > 1:
            #     motion = self.vae.decode(outputs[i])
            # else:
            #     motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        # joints_ref = self.feats2joints(feats_ref)
        # joints_rst = self.feats2joints(feats_rst)        
        if self.cfg.EXPER.motion_repre == 'ske':
            joints_ref = self.feats2joints(feats_ref)
            joints_rst = self.feats2joints(feats_rst)
        else:
            # check if smplx exist in the model
            if not hasattr(self, 'smplx_model'):
                self.get_smplx_model()
            smplx_rst = self.feats2joints(feats_rst)
            smplx_ref = self.feats2joints(feats_ref)
            joints_ref = self.smplx_infer(smplx_ref)
            joints_rst = self.smplx_infer(smplx_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len,
            'pred_valid': pred_valid,
            'code_frequency': None,
            # "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        self.hparams.metrics_dict = []

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        all_captions = batch['all_captions']

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):

            if self.vae_hand != None:
                code_pred_body, _ = self.vae_body.encode(feats_ref[i:i + 1, :lengths[i], ..., :self.vae_body.nfeats])
                code_pred_hand, _ = self.vae_hand.encode(feats_ref[i:i + 1, :lengths[i], ..., self.vae_body.nfeats:])
                code_pred = torch.cat([code_pred_body, code_pred_hand], dim=0)
            else:
                code_pred, _ = self.vae_body.encode(feats_ref[i:i + 1, :lengths[i]])
            
            motion_tokens.append(code_pred)
            lengths_tokens.append(code_pred.shape[1])
            # motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            # motion_tokens.append(motion_token[0])
            # lengths_tokens.append(motion_token.shape[1])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths_tokens,
                                               task="m2t",
                                               stage='test')

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "t_ref": all_captions,
            # "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_inter_forward(self, batch):
        motions = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        m_ref = batch["b_motion"]
        t_ref = batch["b_speech"]
        len_ref = batch["b_length"]
        
        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        
        gt_tokens = []
        
        for i in range(len(motions)):
            motion_token, _ = self.vae.encode(motions[i:i + 1])
            motion_tokens.append(motion_token[0])
            lengths_tokens.append(motion_token.shape[1])
            
            gt_token, _ = self.vae.encode(m_ref[i:i + 1])
            gt_tokens.append(gt_token[0])
        
        output_motion_tokens, output_speech = self.lm.generate_inter(texts, motion_tokens, lengths_tokens, tasks)

        # Motion Decode
        m_rst = torch.zeros_like(m_ref)
        min_len = len_ref.copy()

        # check no motion
        has_motion = []
        for output_motion_token in output_motion_tokens:
            has_motion.append(len(output_motion_token) > 1)
        
        # test vqvae
        # output_motion_tokens = gt_tokens

        for i in range(len(texts)):
            output_motion_tokens[i] = torch.clamp(output_motion_tokens[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(output_motion_tokens[i]) > 1:
                motion = self.vae.decode(output_motion_tokens[i])
            else:
                motion = torch.zeros_like(m_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], min_len[i])

            # Cut Motion
            m_rst[i:i + 1, :min_len[i], ...] = motion[:, :min_len[i]]

        # m_rst = m_ref.clone()
        # Recover joints for evaluation
        joints_ref = self.feats2joints(m_ref)
        joints_rst = self.feats2joints(m_rst)
        joints_input = self.feats2joints(motions)

        # Renorm for evaluation
        m_ref = self.datamodule.renorm4t2m(m_ref)
        m_rst = self.datamodule.renorm4t2m(m_rst)
        m_input = self.datamodule.renorm4t2m(motions)
        

        
        
        # return set
        # TODO here!!!
        # if has_motion.count(True) == 0:
        #     has_motion[0] = True
            
        # rs_set = {
        #     "m_ref": m_ref[has_motion],
        #     "t_ref": [a for a,b in zip(t_ref, has_motion) if b],
        #     "length": [a for a,b in zip(min_len, has_motion) if b],
        #     'm_rst': m_rst[has_motion],
        #     't_pred': [a for a,b in zip(output_speech, has_motion) if b],
        #     'joints_ref': joints_ref[has_motion],
        #     'joints_rst': joints_rst[has_motion]
        # }

        rs_set = {
            "m_ref": m_ref,
            "t_ref": t_ref,
            "length": min_len,
            'm_rst': m_rst,
            't_rst': output_speech,
            'joints_ref': joints_ref,
            'joints_rst': joints_rst,
            'joints_input': joints_input,
            'm_input': m_input,
            't_input': texts,
            'csv_id': batch['csv_id'],
            'length_inputs': batch['length'],
        }

        return rs_set


    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths,
                                               task=task,
                                               stage='test')

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):
        # batch detach
        feats_ref = batch["motion"]
        joints_ref = self.feats2joints(feats_ref)
        # motion encode & decode
        
        if self.vae_hand != None:
            feats_rst_body, loss_commit_body, perplexity_body = self.vae_body(feats_ref[..., :self.vae_body.nfeats])
            feats_rst_hand, loss_commit_hand, perplexity_hand = self.vae_hand(feats_ref[..., self.vae_body.nfeats:])
            feats_rst = torch.cat((feats_rst_body, feats_rst_hand), dim=-1)
            loss_commit = loss_commit_body + loss_commit_hand
            perplexity = perplexity_body + perplexity_hand
        else:
            feats_rst, loss_commit, perplexity = self.vae_body(feats_ref)
        joints_rst = self.feats2joints(feats_rst)
        
        if self.vae_transform != None:
            feats_rst_trans, loss_commit_trans, perplexity_trans = self.vae_transform(batch['transform'])
        else:
            feats_rst_trans, loss_commit_trans, perplexity_trans = None, None, None 
        # return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
            "loss_commit_trans": loss_commit,
            "m_trans_ref": batch['transform'].clone(),
            "m_trans_rst": feats_rst_trans,
        }
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
        # Detach batch
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Repeat for multimodal evaluation
        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        feats_rst = torch.zeros_like(feats_ref)

        # transform_rst = torch.zeros_like(batch['transform'])
        if self.vae_transform != None:
            transform_rst, _, _ = self.vae_transform(batch['transform'])
            code_pred_transform, _ = self.vae_transform.encode(batch['transform'])
            codeFre_pred = torch.bincount(code_pred_transform[:, 0],
                                          minlength=self.vae_transform.code_num).to(
                                              feats_ref.device)
        else:
            codeFre_pred = None
        pred_valid = [True] * len(feats_ref)
        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            
            if self.vae_hand != None:
                feats_pred_body, _, _ = self.vae_body(feats_ref[i:i + 1, :lengths[i], ...,  :self.vae_body.nfeats])
                feats_pred_hand, _, _ = self.vae_hand(feats_ref[i:i + 1, :lengths[i], ...,  self.vae_body.nfeats:])
                feats_pred = torch.cat((feats_pred_body, feats_pred_hand), dim=-1)
                code_pred_body, _ = self.vae_body.encode(feats_ref[i:i + 1, :lengths[i], ..., :self.vae_body.nfeats])
                code_pred_hand, _ = self.vae_hand.encode(feats_ref[i:i + 1, :lengths[i], ..., self.vae_body.nfeats:])
                
            else:
                feats_pred, _, _ = self.vae_body(feats_ref[i:i + 1, :lengths[i]])
                code_pred, _ = self.vae_body.encode(feats_ref[i:i + 1, :lengths[i]])
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred 

            # codeFre_pred = torch.bincount(code_pred[0],
            #                               minlength=self.hparams.codebook_size).to(
            #                                   self.codeFrequency.device)
            # self.codePred.append(code_pred[0])
            # self.codeFrequency += codeFre_pred

        # np.save('../memData/results/codeFrequency.npy',
        #         self.codeFrequency.cpu().numpy())

        # Recover joints for evaluation
        if self.cfg.EXPER.motion_repre == 'ske':
            joints_ref = self.feats2joints(feats_ref)
            joints_rst = self.feats2joints(feats_rst)
        else:
            # check if smplx exist in the model
            if not hasattr(self, 'smplx_model'):
                self.get_smplx_model()
            smplx_rst = self.feats2joints(feats_rst)
            smplx_ref = self.feats2joints(feats_ref)
            joints_ref = self.smplx_infer(smplx_ref)
            joints_rst = self.smplx_infer(smplx_rst)
            
            
        # joints_ref = self.feats2joints(feats_ref)
        # joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "length": lengths,
            'code_frequency': codeFre_pred,
            'pred_valid': pred_valid,
        }

        return rs_set

    def revserse_ric_data(self, data):
        # index from all data to body and hands
        body_index = list(range(0, 4+21*3)) + list(range(4+51*3, 4+51*3+21*6)) + \
                list(range(4+51*9, 4+51*9+22*3)) + list(range(4+51*9+52*3, 4+51*9+52*3+4))
        hand_index = list(range(4+21*3, 4+51*3)) + list(range(4+51*3+21*6, 4+51*9)) + \
            list(range(4+51*9+22*3, 4+51*9+52*3))
            
        all_index_to_feat = body_index + hand_index
        ## get the index of each item in all_index_to_feat
        all_feat_to_ric = {}
        for i, item in enumerate(all_index_to_feat):
            all_feat_to_ric[item] = i
        # sort the dicts by key, ascending
        all_feat_to_ric = dict(sorted(all_feat_to_ric.items(), key=lambda item: item[0]))
        
        reverse_indices = list(all_feat_to_ric.values())
        return data[:, :, reverse_indices]


    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None

        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain"
                                    ] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage == 'lm_rl' and split in ['train']:
            rs_set = self.train_rl_forward(batch)
            loss = None

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl"]:
                if self.hparams.task == "t2m":
                    # TODO check!!!!
                    # return None
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                    rs_set = self.val_m2m_forward(batch, self.hparams.task)
                elif self.hparams.task == "interaction":
                    rs_set = self.val_inter_forward(batch)
                    
            if self.hparams.task not in ["m2t"]:
                # MultiModality evaluation sperately
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')
                
                if batch['motion'].shape[-1] != 263:
                    if 'TM2TMetrics' in metrics_dicts:
                        metrics_dicts.remove('TM2TMetrics')
    
                for metric in metrics_dicts:
                    lengths = batch['length']
                    if metric == "TemosMetric":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "TM2TMetrics":
                        if self.hparams.stage in [
                                "lm_instruct", "lm_pretrain", "lm_rl"
                        ] and self.hparams.task != "interaction":
                            #  and self.hparams.task != "interaction"
                            word_embs = batch['word_embs']
                            pos_ohot = batch['pos_ohot']
                            text_lengths = batch['text_len']
                            # if self.trainer.datamodule.is_mm:
                            #     word_embs = word_embs.repeat_interleave(
                            #         self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                            #         dim=0)
                            #     pos_ohot = pos_ohot.repeat_interleave(
                            #         self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                            #         dim=0)
                            #     text_lengths = text_lengths.repeat_interleave(
                            #         self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                            #         dim=0)
                        else:
                            word_embs = None
                            pos_ohot = None
                            text_lengths = None

                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths,
                        )
                    elif metric == "UncondMetrics":
                        getattr(self.metrics, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=lengths,
                        )
                    elif metric == "MRMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "PredMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "MMMetrics":
                        # pass
                        getattr(self.metrics,
                                metric).update(rs_set["m_rst"],
                                               rs_set['length'])
                    elif metric == 'MTMetrics':
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            pred_valid=rs_set['pred_valid'],
                            code_frequency=rs_set['code_frequency'],
                        )
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            elif self.hparams.task in ["m2t"] and self.hparams.stage in [
                    "lm_instruct", "lm_pretrain", "lm_rl"
            ]:
                self.hparams.metrics_dict = metrics_dicts = ['MTMetrics']
                for metric in metrics_dicts:
                    if metric == "MTMetrics":
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            pred_texts=rs_set["t_pred"],
                            gt_texts=batch["text"],
                            lengths_ref=rs_set['length'],
                        )
        # if split in ['val'] and self.hparams.task == 'interaction':
        #     print("Interaction Evaluation")
        # return forward output rather than loss during test
        if split in ["test"]:
            if self.hparams.task == "t2m":
                return rs_set["joints_rst"], rs_set["length"], rs_set[
                    "joints_ref"]
                # pass
            elif self.hparams.task == "m2t":
                return rs_set["t_pred"], batch["length"]
                # return batch["length"]
            else:
                return rs_set

        return loss
