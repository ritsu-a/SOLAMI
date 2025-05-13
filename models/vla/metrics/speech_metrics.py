import os
import sys
os.environ["TRANSFORMERS_CACHE"] = "~/.cache/huggingface/hub"
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla')
import torch
import numpy as np
from torchmetrics import Metric
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import whisper
from jiwer import wer


class STMetrics(Metric):
    def __init__(self, 
                device='cuda',
                task='t2m',
                vc_feature_extractor=None,
                vc_model=None,
                whisper_model=None,
                dist_sync_on_step=True,
                ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.default_device = device
        self.task = task
        self.metrics = []
        self.add_state("vc_similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("vc_similarity")

        self.add_state("wer", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("wer")
        if vc_feature_extractor is None:
            self.vc_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv').to(self.default_device)
        else:
            self.vc_feature_extractor = vc_feature_extractor
        
        if vc_model is None:
            self.vc_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(self.default_device)
        else:
            self.vc_model = vc_model
    
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_valid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("pred_valid")
        if whisper_model is None:
            self.whisper_model = whisper.load_model('large-v3', download_root='~/.cache/whisper', device=self.default_device)
        else:
            self.whisper_model = whisper_model
        
        self.gt_texts = []
        self.pred_texts = []
    
    def compute_wer(self, target, pred):
        if type(target) is not str:
            target_text = self.whisper_model.transcribe(target)['text']
            pred_text = self.whisper_model.transcribe(pred)['text']
        else:
            target_text = target
            pred_text = pred
        error = wer(target_text, pred_text)
        self.gt_texts.append(target_text)
        self.pred_texts.append(pred_text)
        return error, pred_text
    
    
    def compute_vc_similarity(self, target, pred):
        
        audio = [pred.cpu().numpy(), target.cpu().numpy()]
        inputs = self.vc_feature_extractor(audio, padding=True, return_tensors="pt", sampling_rate=16000)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.default_device)
        with torch.no_grad():
            embeddings = self.vc_model(**inputs).embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
            cosine_sim = torch.nn.CosineSimilarity(dim=-1)
            similarity = cosine_sim(embeddings[0], embeddings[1])
        return similarity
    
    
    def reset(self):
        self.count_seq = torch.tensor(0).to(self.default_device)
        self.pred_valid = torch.tensor(0).to(self.default_device)
        self.wer = torch.tensor(0.0).to(self.default_device)
        self.vc_similarity = torch.tensor(0.0).to(self.default_device)
        self.gt_texts = []
        self.pred_texts = []
    
    
    def log_metrics(self, metrics):
        # for metric in metrics:
        ret_metrics = {}
        if self.task in ['t2s', 's2s']:
            for key in ['wer', 'vc_similarity', 'pred_valid']:
                print(f"{key}: {metrics[key].item()}")
                ret_metrics[key] = metrics[key].item()
        elif self.task in ['s2t']:
            for key in ['wer', 'pred_valid']:
                print(f"{key}: {metrics[key].item()}")
                ret_metrics[key] = metrics[key].item()
        else:
            pass
        return ret_metrics
    
    @torch.no_grad()
    def compute(self,):
        count_seq = self.count_seq.item()
        pred_valid = self.pred_valid.item()
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        if pred_valid == 0:
            self.reset()
            self.gt_texts = []
            self.pred_texts = []
            return {**metrics}
        
        metrics['pred_valid'] = torch.tensor(pred_valid / count_seq).to(self.default_device)
        
        if self.task in ['t2s', 's2s']:
            metrics['wer'] = self.wer / pred_valid
            metrics['vc_similarity'] = self.vc_similarity / pred_valid
        
        if self.task in ['s2t']:
            metrics['wer'] = self.wer / pred_valid
        
        self.reset()
        self.gt_texts = []
        self.pred_texts = []
        
        return {**metrics}
    
    
    @torch.no_grad()
    def update(self, 
               target,       
               pred,
               task: str = 't2s'):
        self.count_seq += 1
        if task in ['t2s', 's2s']:
            if pred is None:
                return None
            else:
                target = target.to(self.default_device)
                pred = pred.to(self.default_device)
                tmp_wer, pred_text = self.compute_wer(target, pred)
                self.wer += tmp_wer
                self.pred_valid += 1
                self.vc_similarity += self.compute_vc_similarity(target, pred)
                return pred_text
        elif task in ['s2t']:
            if pred is None:
                return
            else:
                self.wer += self.compute_wer(target, pred)
                self.pred_valid += 1
        else:
            pass
        pass
        