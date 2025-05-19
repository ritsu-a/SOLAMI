import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/models/vla/anygpt')
os.environ["WANDB_DISABLED"] = "true"
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import logging
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import Trainer
from datasets import load_dataset, interleave_datasets, concatenate_datasets
# from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as DatasetTorch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers import AutoModelForImageTextToText, TorchAoConfig, Gemma3ForConditionalGeneration, AutoProcessor, AutoTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint

from m_utils.loggings import get_logger
from m_utils.prompter import *
from m_utils.anything2token import *


DEBUG=False
LENGTHS = 6400

class SpeechDataset(DatasetTorch):
    def __init__(self,
                 data_path: str,
                tokenizer: transformers.PreTrainedTokenizer,
                data_args,
                logger,
                prompter,
                raw_dataset=None,):
        super(SpeechDataset, self).__init__()
        if raw_dataset is not None:
            self.raw_dataset = raw_dataset
        else:
            logger.info("Loading from dataset {}".format(data_path.split('/')[-1]))
            raw_dataset = load_dataset("json", data_files=data_path)
            raw_dataset = raw_dataset['train']
            if DEBUG:
                raw_dataset = raw_dataset.select(range(LENGTHS))
            self.raw_dataset = raw_dataset
        self.logger = logger
        logger.info("Loading {} items from dataset {}".format(len(self.raw_dataset), data_path.split('/')[-1]))
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.prompter = prompter

    def __len__(self):
        return len(self.raw_dataset)

    @staticmethod
    def train_test_split(dataset, test_size=100, shuffle=True, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)

        if test_size < 1:
            test_size = int(test_size * len(dataset))
        elif test_size >= len(dataset)-32:
            test_size = len(dataset) - 32
        split = len(dataset) - test_size
        train_indices = indices[:split]
        test_indices = indices[split:]

        raw_dataset_train = dataset.raw_dataset.select(train_indices)
        raw_dataset_test = dataset.raw_dataset.select(test_indices)
        train_data = SpeechDataset('', raw_dataset=raw_dataset_train, tokenizer=dataset.tokenizer, data_args=dataset.data_args, logger=dataset.logger, prompter=dataset.prompter)
        test_data = SpeechDataset('', raw_dataset=raw_dataset_test, tokenizer=dataset.tokenizer, data_args=dataset.data_args, logger=dataset.logger, prompter=dataset.prompter)

        return {'train': train_data, 'test': test_data}

    def tokenize_func(self, sentence, add_eos_token=True):
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.tokenizer.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def __getitem__(self, i):
        raw_data = self.raw_dataset[i]
        chat_data = raw_data['chat']
        if len(chat_data) > 1:
            tasks = ['t2s', 's2t', 's2s']
        elif len(chat_data) == 1:
            tasks = ['t2s', 's2t']
        else:
            self.logger.error("invalid chat data at {}".format(raw_data['id']))

        ### random choice the task for this sample using torch
        task = random.choice(tasks)
        if task  == 's2s':
            chat_idx = random.choice(range(len(chat_data)//2))
            # no interleave
            speech1 = chat_data[chat_idx * 2]['speech']
            speech2 = chat_data[chat_idx * 2 + 1]['speech']

            res = self.prompter.generate_x2x_template(
                modality1_str=speech1,
                modality2_str=speech2,
                modality="speech"
            )
        else:
            chat_idx = random.choice(range(len(chat_data)))
            text = chat_data[chat_idx]['text']

            speech = chat_data[chat_idx]['speech']

            if task == 't2s':
                res = self.prompter.generate_t2x_template(
                    modality_str=speech,
                    text=text,
                    modality="speech"
                )
            else:
                res = self.prompter.generate_x2t_template(
                    modality_str=speech,
                    text=text,
                    modality="speech"
                )
        result = self.tokenize_func(res)
        return result


class ITDataset(DatasetTorch):
    def __init__(self,
                data_path: str,
                tokenizer: transformers.PreTrainedTokenizer,
                data_args,
                logger,
                prompter,
                raw_dataset=None,):
        super(ITDataset, self).__init__()
        if raw_dataset is not None:
            self.raw_dataset = raw_dataset
        else:
            logger.info("Loading from dataset {}".format(data_path.split('/')[-1]))
            raw_dataset = load_dataset("json", data_files=data_path)
            raw_dataset = raw_dataset['train']
            # TODO debug
            if DEBUG:
                raw_dataset = raw_dataset.select(range(LENGTHS))
            self.raw_dataset = raw_dataset
        self.logger = logger
        logger.info("Loading {} items from dataset {}".format(len(self.raw_dataset), data_path.split('/')[-1]))
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.prompter = prompter

    def __len__(self):
        return len(self.raw_dataset) * 10

    @staticmethod
    def train_test_split(dataset, test_size=100, shuffle=True, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        # print('len(dataset it):', len(dataset.raw_dataset))
        indices = list(range(len(dataset.raw_dataset)))
        if shuffle:
            random.shuffle(indices)

        if test_size < 1:
            test_size = int(test_size * len(dataset))
        elif test_size >= len(dataset)-32:
            test_size = len(dataset) - 32

        test_raw_size = test_size // 10
        # print('test_raw_size:', test_raw_size)
        split = len(dataset.raw_dataset) - test_raw_size
        train_indices = indices[:split]
        test_indices = indices[split:]

        raw_dataset_train = dataset.raw_dataset.select(train_indices)
        raw_dataset_test = dataset.raw_dataset.select(test_indices)
        train_data = ITDataset('', raw_dataset=raw_dataset_train, tokenizer=dataset.tokenizer, data_args=dataset.data_args, logger=dataset.logger, prompter=dataset.prompter)
        test_data = ITDataset('', raw_dataset=raw_dataset_test, tokenizer=dataset.tokenizer, data_args=dataset.data_args, logger=dataset.logger, prompter=dataset.prompter)

        return {'train': train_data, 'test': test_data}

    def tokenize_func(self, sentence, add_eos_token=True):
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.tokenizer.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def __getitem__(self, i):
        idx = i // 10
        round_id = i % 10
        raw_data = self.raw_dataset[idx]
        chat_data = raw_data['chat']
        if True:
            tasks = ['t2s', 's2t']
        # else:
        #     tasks = ['t2s', 's2t', 's2s']
        ### random choice the task for this sample using torch
        task = random.choice(tasks)
        if task  == 's2s':
            # no interleave
            speech1 = chat_data[round_id]['speech']
            speech2 = chat_data[round_id + 1]['speech']
            speech1 = modality_tokens_to_string(speech1, modality="speech")
            speech2 = modality_tokens_to_string(speech2, modality="speech")
            res = self.prompter.generate_x2x_template(
                modality1_str=speech1,
                modality2_str=speech2,
                modality="speech"
            )
        else:

            text = chat_data[round_id]['speech_text']

            speech = chat_data[round_id]['speech']
            speech = modality_tokens_to_string(speech, modality="speech")
            if task == 't2s':
                res = self.prompter.generate_t2x_template(
                    modality_str=speech,
                    text=text,
                    modality="speech"
                )
            else:
                res = self.prompter.generate_x2t_template(
                    modality_str=speech,
                    text=text,
                    modality="speech"
                )
        result = self.tokenize_func(res)
        return result


class MotionDataset(DatasetTorch):
    def __init__(self,
                 data_path: str,
                tokenizer: transformers.PreTrainedTokenizer,
                data_args,
                logger,
                prompter,
                raw_dataset=None,):
        super(MotionDataset, self).__init__()
        if raw_dataset is not None:
            self.raw_dataset = raw_dataset
        else:
            raw_dataset = load_dataset("json", data_files=data_path)
            raw_dataset = raw_dataset['train']
            # TODO debug
            if DEBUG:
                raw_dataset = raw_dataset.select(range(LENGTHS))
            self.raw_dataset = raw_dataset
        self.logger = logger
        logger.info("Loaded {} items from dataset {}".format(len(self.raw_dataset), data_path.split('/')[-1]))
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.prompter = prompter

    def __len__(self):
        return len(self.raw_dataset)

    @staticmethod
    def train_test_split(dataset, test_size=100, shuffle=True, random_state=None):
        if random_state is not None:
            random.seed(random_state)

        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)

        if test_size < 1:
            test_size = int(test_size * len(dataset))
        elif test_size >= len(dataset)-32:
            test_size = len(dataset) - 32
        split = len(dataset) - test_size
        train_indices = indices[:split]
        test_indices = indices[split:]

        raw_dataset_train = dataset.raw_dataset.select(train_indices)
        raw_dataset_test = dataset.raw_dataset.select(test_indices)
        train_data = MotionDataset('', raw_dataset=raw_dataset_train, tokenizer=dataset.tokenizer, data_args=dataset.data_args, logger=dataset.logger, prompter=dataset.prompter)
        test_data = MotionDataset('', raw_dataset=raw_dataset_test, tokenizer=dataset.tokenizer, data_args=dataset.data_args, logger=dataset.logger, prompter=dataset.prompter)

        return {'train': train_data, 'test': test_data}

    def tokenize_func(self, sentence, add_eos_token=True):
        result = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.tokenizer.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def __getitem__(self, i):
        raw_data = self.raw_dataset[i]
        chat_data = raw_data

        if False:
            tasks = ['t2m', 'm2t', 'm2m']
        elif True:
            tasks = ['t2m', 'm2t']
        else:
            self.logger.error("invalid chat data at {}".format(raw_data['id']))

        ### random choice the task for this sample using torch
        task = random.choice(tasks)
        if task  == 'm2m':
            chat_idx = random.choice(range(len(chat_data)//2))
            # no interleave

            body1 = chat_data[chat_idx * 2]['body']
            hand1 = chat_data[chat_idx * 2]['hand']
            trans1 = chat_data[chat_idx * 2]['trans']
            body2 = chat_data[chat_idx * 2 + 1]['body']
            hand2 = chat_data[chat_idx * 2 + 1]['hand']

            motion1 = modality_tokens_to_string(trans1, modality="trans") + \
                modality_tokens_to_string(body1, modality="body") + \
                    modality_tokens_to_string(hand1, modality="hand")

            motion2 = modality_tokens_to_string(body2, modality="body") + \
                    modality_tokens_to_string(hand2, modality="hand")

            res = self.prompter.generate_x2x_template(
                modality1_str=motion1,
                modality2_str=motion2,
                modality="motion"
            )
        else:
            text = random.choice(chat_data['text'])

            motion = chat_data['motion']

            if task == 't2m':
                motion = modality_tokens_to_string(motion, modality="motion") 
                res = self.prompter.generate_t2x_template(
                    modality_str=motion,
                    text=text,
                    modality="motion"
                )
            else:
                motion = modality_tokens_to_string(motion, modality="motion")
                res = self.prompter.generate_x2t_template(
                    modality_str=motion,
                    text=text,
                    modality="motion"
                )
        result = self.tokenize_func(res)
        return result


class WeightedDataset(DatasetTorch):
    def __init__(self, datasets, ratios):
        self.datasets = datasets
        self.dataset_list = []
        for key in ['motion', 'speech']:
            self.dataset_list += datasets[key]
        assert sum(ratios) == 1
        self.ratios = ratios
        self.total_length = int(sum([len(dataset) for dataset in datasets['motion']]) / ratios[0])
        # print('total_length:', self.total_length)
        self.count = 0

    def __len__(self):
        return self.total_length


    def __getitem__(self, index):
        self.count += 1

        ### warm up ration of motion
        data_warm_up_ratio = 0.5
        motion_weight = self.count / (data_warm_up_ratio * self.total_length)
        motion_weight = min(motion_weight, 1)
        new_ratios = [ratio for ratio in self.ratios]
        new_ratios[0] *=  motion_weight
        sum_ratio = sum(new_ratios)
        new_ratios = [ratio / sum_ratio for ratio in new_ratios]

        ### sample dataset
        weights_len = []
        motion_len = self.total_length * new_ratios[0]
        motion_weight_lens = [len(dataset) for dataset in self.datasets['motion']]
        motion_weight_lens = [x / sum(motion_weight_lens) for x in motion_weight_lens]
        motion_weight_lens = [x * motion_len for x in motion_weight_lens]
        weights_len += motion_weight_lens

        speech_len = self.total_length * new_ratios[1]
        speech_weight_lens = [len(dataset) for dataset in self.datasets['speech']]
        speech_weight_lens = [x / sum(speech_weight_lens) for x in speech_weight_lens]
        speech_weight_lens = [x * speech_len for x in speech_weight_lens]
        weights_len += speech_weight_lens

        weights_norm = [x / sum(weights_len) for x in weights_len]

        sampled_dataset = random.choices(self.dataset_list, weights=weights_norm, k=1)[0]
        data_id = index % len(sampled_dataset)
        return sampled_dataset[data_id]