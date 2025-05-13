import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen')
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import logging
import json
from llm_api import calculate_api_cost, get_embedding
from omegaconf import OmegaConf
from datasets.dataset import get_datasets, check_allowed_actions, get_dataset_item_weights
from base_prompt import *
from generation import generate_by_scripts_completion, generate_by_agent_conversation, generate_by_generate_once
from logger import get_logger

TOKENS_BLOG = {}
logging.basicConfig(level=logging.INFO)


def get_prompts_from_profile(profile, profile_settings, logger=None, **kwargs):
    prompts = {}
    Nan = -1
    prompts['background'] = BACKGROUND
    
    if kwargs['content_type'] == 'motion imitation':
        prompts['background'] += get_motion_imitation_prompt()
    elif kwargs['content_type'] == 'instruction following':
        prompts['background'] += get_instruction_following_prompt()
    elif kwargs['content_type'] == 'motion understanding':
        prompts['background'] += get_motion_understanding_prompt()
    elif kwargs['content_type'] == 'common':
        prompts['background'] += ''
    else:
        raise ValueError("Invalid content type")
    
    prompts['user_settings'] = USER_SETTINGS
    
    if profile[0] == Nan or profile[0] is None:
        prompts['agent_settings'] = AGENT_SETTINGS['assistant']
    elif profile[0] in AGENT_SETTINGS.keys():
        prompts['agent_settings'] = AGENT_SETTINGS[profile[0]]
    else:
        logger.info("Invalid agent settings, use default assistant settings")
        prompts['agent_settings'] = AGENT_SETTINGS['assistant']
    
    if profile[1] == Nan or profile[1] is None:
        prompts['topic_type'] = profile_settings['topic_type']
        prompts['topic'] = ''
    else:
        prompts['topic_type'] = 'preset'
        prompts['topic'] = profile[1]
    
    prompts['space_limitation'] = SPACE_LIMITATION
    prompts['locomotion_limitation'] = LOCOMOTION
    prompts['behavior_illustration'] = BEHAVIOR_ILLUSTRATION
    
    return prompts


def generate_scripts(
    profile=[],
    datasets=None,
    tokens_blog={},
    logger=None,
    **kwargs,
):
    ## check presetting
    # row['Background'], row['User Settings'], row['Agent Settings'], row['Topic']
    prompts = get_prompts_from_profile(profile, logger=logger, **kwargs)
    
    ## generate conversation scripts
    if kwargs['conversation_settings']['method'] == 'script completion':
        generation_res = generate_by_scripts_completion(
            prompts=prompts,
            datasets=datasets,
            tokens_blog=tokens_blog,
            logger=logger,
            **kwargs,
        )
    elif kwargs['conversation_settings']['method'] == 'agent conversation':
        generation_res = generate_by_agent_conversation(
            prompts=prompts,
            datasets=datasets,
            tokens_blog=tokens_blog,
            logger=logger,
            **kwargs,
        )
    elif kwargs['conversation_settings']['method'] == 'generate once':
        generation_res = generate_by_generate_once(
            prompts=prompts,
            datasets=datasets,
            tokens_blog=tokens_blog,
            logger=logger,
            **kwargs,
        )
    
    return generation_res


def save_res(generate_res, output_path):
    if output_path.endswith('.csv'):
        pass
    elif output_path.endswith('.json'):
        dialogs_dict = {}
        for index, behavior in enumerate(generate_res['dialogs_raw']):
            dialogs_dict[index] = behavior.get_save_dicts()
        generate_res['dialogs'] = dialogs_dict
        generate_res.pop('dialogs_raw')
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                data.append(generate_res)
            else:
                raise ValueError("Invalid save format")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                json.dump([generate_res], f, indent=2)
        logging.info(f"Save to {output_path}!")
    else:
        raise ValueError("Invalid save format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_data_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default='/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/configs/default.yaml')
    parser.add_argument("--exper", type=str, default='debug-5')
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    args.profile_data_path = '/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data/topics/merged_topics.csv'
    
    params =  OmegaConf.load(args.config_path)
    api_info = OmegaConf.load("/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/configs/api.yaml")
    
    if params.llm_settings.api_type in api_info.API_info.keys():
        params.llm_settings.update(api_info.API_info[params.llm_settings.api_type])
    if params.text_embedding_settings.api_type in api_info.API_info.keys():
        params.text_embedding_settings.update(api_info.API_info[params.text_embedding_settings.api_type])
    
    
    if args.output_dir is not None:
        params['save_settings']['save_path'] = args.output_path
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
    else:
        if not os.path.exists(params['save_settings']['save_path']):
            os.makedirs(params['save_settings']['save_path'])
    
    output_path = os.path.join(params['save_settings']['save_path'], f'{args.exper}.{params["save_settings"]["save_format"]}')
    logger_path = os.path.join(params['save_settings']['save_path'], params['logging_settings']['log_file'])
    logger = get_logger(save_path=logger_path, **params['logging_settings'])
    
    ## get datasets
    # text_to_actions_dicts, text_embeddings_dicts = get_datasets(params['dataset_names'], TOKENS_BLOG, params)
    # check_allowed_actions(text_to_actions_dicts, text_embeddings_dicts, params['interaction_allowed'])
    # dataset_item_weights = get_dataset_item_weights(text_to_actions_dicts, params['dataset_weights'])
    # params['dataset_item_weights'] = dataset_item_weights
    datasets = get_datasets(params['dataset_names'], data_configs=params['datasets'], logger=logger)
    
    ## get profile_data for generation
    profile_data = []
    if args.profile_data_path is not None:
        data_csv = pd.read_csv(args.profile_data_path)
        data_csv.fillna(-1, inplace=True)
        for index, row in data_csv.iterrows():
            profile_data.append([row['Agent Settings'], row['modified_topic']])
    else:
        profile_data.append(['', ''])
    
    
    ## generate scripts
    data_generated = []
    params.pop('datasets')
    for profile in profile_data[1500::80]:
        for repeat in range(params['conversation_settings']['NUM_EXAMPLES']):
            dialogs_info = generate_scripts(
                profile=profile,
                datasets=datasets,
                tokens_blog=TOKENS_BLOG,
                logger=logger,
                **params,
            )
            if dialogs_info == {}:
                continue
            dialogs_info['repeat'] = repeat
            
            for dialog in dialogs_info['dialogs_raw']:
                print(dialog)
            
            save_res(dialogs_info, output_path)
            data_generated.append(dialogs_info)
    
    ### Final
    cost_all, costs = calculate_api_cost(TOKENS_BLOG)
    print('Total cost ($): ', cost_all)
    print('Detail costs ($): ', costs)
    pass