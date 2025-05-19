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

import torch.distributed as dist
import debugpy

def initialize_debugpy():
    # if not dist.is_initialized() and dist.get_rank() == 0:
        # print(f"Rank: {dist.get_rank()} - Debugpy is listening on port 15696")
        print("Debugpy is listening on port 15696")
        debugpy.listen(("0.0.0.0", 15696))
        debugpy.wait_for_client()

def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

# initialize_distributed()
# initialize_debugpy()



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="share_data/llama2_hf/llama-2-7b-hf",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

@dataclass
class DataArguments:
    speech_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training speech data."}
    )
    motion_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training motion data."}
    )
    it_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training instruction tuning data."}
    )
    cache_dir: Optional[str] = field(
        default="mllm/data/both_cache",
        metadata={"help": "Where do you want to store the tokenized data"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "block_size"
            )
        },
    )
    concatenating: bool = field(
        default=True,
        metadata={"help": "Enable concatenating mode"}
    )
    preprocessing_num_workers: int = field(
        default=50,
        metadata={"help": "preprocessing_num_workers for tokenizing"},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "use_flash_attn"},
    )
    val_set_size: int = field(
        default=1000,
        metadata={"help": "val_set_size"},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "evaluation_strategy"},
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "eval_steps"},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "save_strategy"},
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "save_steps"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "num_epochs"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "learning_rate"},
    )
    output_dir: str = field(
        default="",
        metadata={"help": "output_dir"},
    )
    train_on_inputs: bool = field(
        default=True,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    initial_global_step: int = field(
        default=0,
        metadata={"help": "initial_global_step"}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "initial_global_step"}
    )
    only_train_new_embeds: bool = field(
        default=False,
        metadata={"help": "only_train_new_embeds"}
    )
    run_name: str = field(
        default="no run name :)",
        metadata={"help": "run_name"}
    )
    log_on_each_node: bool = field(
        default=False,
        metadata={"help": "log_on_each_node"}
    )


class SpeechDataset(DatasetTorch):
    def __init__(self,
                 data_path: str,
                tokenizer: transformers.PreTrainedTokenizer,
                data_args: DataArguments,
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
                data_args: DataArguments,
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
                data_args: DataArguments,
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
        chat_data = raw_data['chat']

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
            chat_idx = 0
            text = random.choice(chat_data[chat_idx]['text'])

            body = chat_data[chat_idx]['body']
            hand = chat_data[chat_idx]['hand']
            trans = chat_data[chat_idx]['trans']

            if task == 't2m':
                motion = modality_tokens_to_string(body, modality="body") + \
                    modality_tokens_to_string(hand, modality="hand")
                res = self.prompter.generate_t2x_template(
                    modality_str=motion,
                    text=text,
                    modality="motion"
                )
            else:
                motion = modality_tokens_to_string(trans, modality="trans") + \
                modality_tokens_to_string(body, modality="body") + \
                modality_tokens_to_string(hand, modality="hand")
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


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger = get_logger(local_rank=training_args.local_rank,
                        save_path=os.path.join(training_args.output_dir, 'train.log'),
                        log_level='debug')

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    prompter = Prompter()

    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-pt', trust_remote_code=True)
  
    tokenizer.padding_side = "left"  # Allow batched inference
    # for token in [user_name, chatbot_name, user_end, chatbot_end]:
    #     if token not in tokenizer.get_vocab():
    #         logger.info(f"Add special unit tokens {token} to tokenizer.vocab")
    #         tokenizer.add_tokens([token])

    for modality in modal_special_str.keys():
        prefix=modal_special_str[modality]["prefix"]
        start=modal_special_str[modality]["sos"]
        end=modal_special_str[modality]["eos"]
        modality_vocab_size = modal_special_str[modality]["vocab_size"]
        if start not in tokenizer.get_vocab():
            logger.info(f"Add {modality} tokens <{prefix}0>-<{prefix}{modality_vocab_size-1}> to tokenizer.vocab")
            tokens = [f"<{prefix}{x}>" for x in range(modality_vocab_size)]
            if start != '':
                tokens += [start, end]
            tokenizer.add_tokens(tokens, True)

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # Define model init arguments
    model_kwargs = dict(
        attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
        # device_map="auto", # Let torch decide how to load the model
    )

    # BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    # Load model and tokenizer
    model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-pt", **model_kwargs)
 

    # resize embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # model = None


    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 4096:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 4096
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)


    if data_args.speech_data_path is not None:
        speech_data_paths = data_args.speech_data_path.split(" ")
        speech_train_datasets = []
        speech_val_datasets = []
        speech_dataset_names = []
        for speech_data_path in speech_data_paths:
            speech_dataset = SpeechDataset(speech_data_path, tokenizer, data_args, logger, prompter)
            if training_args.val_set_size > 0:
                train_val = SpeechDataset.train_test_split(speech_dataset, test_size=training_args.val_set_size, shuffle=True, random_state=42)
                val_data = train_val["test"]
                train_data = train_val["train"]
            else:
                val_data = None
                train_data = speech_dataset
            # train_data, val_data = load_and_preprocess(speech_data_path, modality="speech")
            logger.info("Train data {}  {}".format(speech_data_path, len(train_data)))
            speech_train_datasets.append(train_data)
            speech_val_datasets.append(val_data)
            speech_dataset_names.append(speech_data_path.split("/")[-1].split(".")[0])
        # speech_train_data = WeightedDataset(speech_train_datasets, [len(dataset_) for dataset_ in speech_train_datasets])
        # speech_train_data = speech_train_data.shuffle(seed=42)
        speech_val_data = {}
        for speech_dataset_name, speech_val_dataset in zip(speech_dataset_names, speech_val_datasets):
            speech_val_data[speech_dataset_name] = speech_val_dataset

    if data_args.motion_data_path is not None:
        motion_data_paths = data_args.motion_data_path.split(" ")
        motion_train_datasets = []
        motion_val_datasets = []
        motion_val_dataset_names = []
        for motion_data_path in motion_data_paths:
            print(motion_data_path)
            motion_dataset = MotionDataset(motion_data_path, tokenizer, data_args, logger, prompter)
            motion_file_name = motion_data_path.split('/')[-1]
            if 'test' in motion_file_name:
                if training_args.val_set_size > 0:
                    train_val = MotionDataset.train_test_split(motion_dataset, test_size=training_args.val_set_size, shuffle=True, random_state=42)
                    val_data = train_val["test"]
                    train_data = train_val["train"]
            else:
                train_data = motion_dataset
                val_data = None

            if train_data is not None:
                motion_train_datasets.append(train_data)
            if val_data is not None:
                motion_val_datasets.append(val_data)
                motion_val_dataset_names.append(motion_data_path.split("/")[-1].split(".")[0])
        motion_val_data = {}
        for motion_dataset_name, motion_val_dataset in zip(motion_val_dataset_names, motion_val_datasets):
            motion_val_data[motion_dataset_name] = motion_val_dataset

    # if data_args.it_data_path is not None:
    #     it_data_paths = data_args.it_data_path.split(" ")
    #     it_train_datasets = []
    #     it_val_datasets = []
    #     it_val_dataset_names = []
    #     for it_data_path in it_data_paths:
    #         it_dataset = ITDataset(it_data_path, tokenizer, data_args, logger, prompter)
    #         it_file_name = it_data_path.split('/')[-1]
    #         if training_args.val_set_size > 0:
    #             train_val = ITDataset.train_test_split(it_dataset, test_size=training_args.val_set_size, shuffle=True, random_state=42)
    #             val_data = train_val["test"]
    #             train_data = train_val["train"]
    #         else:
    #             val_data = None
    #             train_data = it_dataset

    #         if train_data is not None:
    #             it_train_datasets.append(train_data)
    #         if val_data is not None:
    #             it_val_datasets.append(val_data)
    #             it_val_dataset_names.append(it_data_path.split("/")[-1].split(".")[0])
    #     it_val_data = {}
    #     for it_dataset_name, it_val_dataset in zip(it_val_dataset_names, it_val_datasets):
    #         it_val_data[it_dataset_name] = it_val_dataset



    if data_args.speech_data_path is not None and data_args.motion_data_path is not None:
        train_datasets = {
            'motion': motion_train_datasets,
            'speech': speech_train_datasets,
        }
        ratios = [0.4, 0.6]
        train_data = WeightedDataset(train_datasets, ratios)
        val_data_dict = {}
        val_data_dict.update(motion_val_data)
        val_data_dict.update(speech_val_data)
        val_data = val_data_dict
        pass
    elif data_args.speech_data_path is not None:
        train_data = WeightedDataset(speech_train_datasets, [0.5, 0.5])
        val_data = speech_val_data
    else:
        exception_str = "motion_data_path and speech_data_path cannot be both None"
        logger.error(exception_str)


    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    logger.info(f"start training")


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=val_data if training_args.do_eval else None,
        data_collator=data_collator
    )

    if training_args.initial_global_step != 0:
        logger.info(f"Set initial global step={training_args.initial_global_step}")
        trainer.state.global_step = training_args.initial_global_step

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()