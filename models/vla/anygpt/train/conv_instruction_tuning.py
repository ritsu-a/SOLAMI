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
import copy
import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


from datasets import load_dataset, interleave_datasets, concatenate_datasets

from torch.utils.data import Dataset as DatasetTorch
from transformers import LlamaForCausalLM, LlamaTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from m_utils.loggings import get_logger
from m_utils.prompter import *
from m_utils.anything2token import *
from m_utils.conversation import get_conv_template


IGNORE_TOKEN_ID=-100

DEBUG=False
LENGTHS = 128

import torch.distributed as dist
import debugpy

def initialize_debugpy():
    if not dist.is_initialized() and dist.get_rank() == 0:
        # print(f"Rank: {dist.get_rank()} - Debugpy is listening on port 15696")
        print("Debugpy is listening on port 15696")
        debugpy.listen(("0.0.0.0", 15696))
        debugpy.wait_for_client()
        
def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

# initialize_distributed()
# initialize_debugpy()

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
#     """Collects the state dict and dump to disk."""
#     if trainer.deepspeed:
#         torch.cuda.synchronize()
#         # trainer.save_model(output_dir)
#         # Save the model directly without invoking trainer.save_model
#         # This prevents creation of `global_step` folders and unnecessary DeepSpeed files.
#         trainer.model.save_pretrained(output_dir, safe=True)
        
#         # Optionally, you can save the tokenizer as well
#         if trainer.tokenizer:
#             trainer.tokenizer.save_pretrained(output_dir)
#         return
#     state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict)
        
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
    lora_r: int = field(
        default=8,
        metadata={
            "help": (
                "loar rank"
            )
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={
            "help": (
                "loar alpha"
            )
        },
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={
            "help": (
                "loar dropout"
            )
        },
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj",
        metadata={
            "help": (
                "lora target modules"
            )
        }
    )


@dataclass
class DataArguments:
    it_data_path: str = field(
        default=None,
        metadata={"help": "it_data_path"},
    )
    cache_dir: Optional[str] = field(
        default="/mnt/petrelfs/zhanjun.p/mllm/data/cache/sft",
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
        default=26,
        metadata={"help": "preprocessing_num_workers for tokenizing"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_lora: bool = field(
        default=True,
        metadata={"help": "use_lora"},
    )
    it_type: str = field(
        default="qlora fsdp int8",
        metadata={"help": "it_type"},
    )
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
        return len(self.raw_dataset)

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
            
        split = len(dataset.raw_dataset) - test_size
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

    def __getitem__(self, idx):
        pass
        raw_data = self.raw_dataset[idx]
        raw_chat_data = raw_data['chat']

        start_round = random.choice([0,1,]) * 2
        last_round = random.choice([2, 3, 4, 5]) * 2
        end_round = min(len(raw_chat_data), start_round + last_round)
        user = None
        agent = None
        conv = get_conv_template("SOLAMI")
        conv.reset()
        user_name_ = "[Human]"
        chatbot_name_ = "[MMGPT]"
        for round_id in range(start_round, end_round):
            tmp_chat_data = raw_chat_data[round_id]
            speech = tmp_chat_data['speech']
            speech_str = modality_tokens_to_string(speech, modality="speech")
            if round_id % 2 == 0:
                motion_str = modality_tokens_to_string(tmp_chat_data['trans'], modality="trans") +\
                    modality_tokens_to_string(tmp_chat_data['body'], modality="body") +\
                    modality_tokens_to_string(tmp_chat_data['hand'], modality="hand")
                user = tmp_chat_data['role']
            else:
                motion_str = modality_tokens_to_string(tmp_chat_data['body'], modality="body") +\
                    modality_tokens_to_string(tmp_chat_data['hand'], modality="hand")
                agent = tmp_chat_data['role']

            ### motion first, then speech
            message_str = motion_str + speech_str
            if tmp_chat_data['role'] == 'User':
                conv.append_message(user_name_, message_str)
            else:
                conv.append_message(chatbot_name_, message_str)
            
        input_conv_str = conv.get_prompt(agent_role=agent)
        
        result = self.tokenizer(
            input_conv_str,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = result['input_ids']
        attention_mask = result['attention_mask']
        
        if (input_ids[-1] != self.tokenizer.eos_token_id and len(input_ids) < self.tokenizer.model_max_length):
                input_ids.append(self.tokenizer.eos_token_id)
                attention_mask.append(1)
        target = copy.deepcopy(input_ids)
        
        sep = conv.sep + '\n' + conv.roles[1] + ": "
        total_len = len(target)
        turns = input_conv_str.split(conv.sep2)
        cur_len = 1

        target[0] = IGNORE_TOKEN_ID        
        for i, turn in enumerate(turns):
            if cur_len >= self.tokenizer.model_max_length:
                break
            if turn == "":
                break
            turn_len = len(self.tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not self.tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            for k in range(cur_len, min(cur_len+instruction_len, self.tokenizer.model_max_length)):
                target[k] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not self.tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1
        excepted_len = total_len - 1
        for k in range(cur_len, excepted_len):
            target[k] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = torch.tensor(target).clone()
            z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
            print(self.tokenizer.decode(z))
            # exit()

        if cur_len < self.tokenizer.model_max_length:
            if cur_len != excepted_len:
                for k in range(total_len):
                    target[k] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target,
        )
    


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
    
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    for token in [user_name, chatbot_name, user_end, chatbot_end]:
        if token not in tokenizer.get_vocab():
            logger.info(f"Add special unit tokens {token} to tokenizer.vocab")
            tokenizer.add_tokens([token])
    
    for modality in modal_special_str.keys():
        prefix=modal_special_str[modality]["prefix"]
        start=modal_special_str[modality]["sos"]
        end=modal_special_str[modality]["eos"]
        modality_vocab_size = modal_special_str[modality]["vocab_size"]
        if start not in tokenizer.get_vocab():
            if start != '':
                tokens = [start, end]
            if f"<{prefix}0>" not in tokenizer.get_vocab():
                logger.info(f"Add {modality} tokens <{prefix}0>-<{prefix}{modality_vocab_size-1}> to tokenizer.vocab")
                tokens += [f"<{prefix}{x}>" for x in range(modality_vocab_size)]
                tokenizer.add_tokens(tokens)
            pass
    
    training_types = training_args.it_type
    
    if training_types == "qlora int4 deepspeed":
        bnb_model_from_pretrained_args = {}
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # load_in_8bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.float16,
            )
        ))
        
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
            **bnb_model_from_pretrained_args,
        )
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        
        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(','),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    elif training_types == "lora deepspeed":
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
        )
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(','),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    elif training_types == "qlora fsdp int8":
        
        bnb_model_from_pretrained_args = {}
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            )
        ))

        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
            load_in_8bit=True,
        )
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        
        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(','),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    elif training_types == 'full':
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
        )
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Invalid training types: {training_types}")

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if data_args.it_data_path is not None:
        it_data_paths = data_args.it_data_path.split(' ')
        it_train_datasets = []
        it_val_datasets = []
        it_val_dataset_names = []
        for it_data_path in it_data_paths:
            it_data = ITDataset(it_data_path, tokenizer, data_args, logger, prompter)
            it_file_name = it_data_path.split('/')[-1]
            if 'train' in it_file_name:
                it_train_datasets.append(it_data)
            elif 'test' in it_file_name:
                it_val_datasets.append(it_data)
                it_val_dataset_names.append(it_file_name)
            else:
                raise ValueError(f"Invalid data path: {it_data_path}")
        it_val_data = {}
        for it_val_dataset_name, it_val_dataset in zip(it_val_dataset_names, it_val_datasets):
            it_val_data[it_val_dataset_name] = it_val_dataset
    else:
        raise ValueError("Invalid data path")

    train_data = it_train_datasets[0]
    val_data = it_val_data
    
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
    
    print('trainer.args.use_lora: ', trainer.args.use_lora)
    
    if training_args.initial_global_step != 0:
        logger.info(f"Set initial global step={training_args.initial_global_step}")
        trainer.state.global_step = training_args.initial_global_step
        
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint_name = os.path.join(training_args.resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(training_args.resume_from_checkpoint, "adapter_model.bin")
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                model = set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
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
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias='none')

if __name__ == '__main__':
    train()