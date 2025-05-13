import os
import sys
sys.path.append('/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen')
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import json
from llm_api import calculate_api_cost
from omegaconf import OmegaConf
from datasets.dataset import get_datasets
from base_prompt import *
from generation import generate_by_scripts_completion, generate_by_agent_conversation, generate_by_generate_once
from logger import get_logger
import random


TOKENS_BLOG = {}

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
        raise logger.critical("Invalid content type %s", kwargs['content_type'])
    
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
    
    generation_res = {}
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
    else:
        logger.critical("Invalid conversation method")
    
    return generation_res


def save_res(generate_res, output_path, logger=None):
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
                raise logger.critical("Invalid save format")
                return 
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                json.dump([generate_res], f, indent=2)
        logger.critical(f"Save to {output_path}!")
    else:
        raise ValueError("Invalid save format")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_data_path", type=str, default='/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data/topics/merged_topics_new.csv')
    parser.add_argument("--config_path", type=str, default='/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/configs/default.yaml')
    parser.add_argument("--exper", type=str, default='sim_all')
    parser.add_argument("--period", type=int, default=8)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    debug = False
    if debug:
        shift = 3680
        args.period = 120
    else:
        shift = 0
    
    params =  OmegaConf.load(args.config_path)
    api_info = OmegaConf.load("/root/pengyang/codebase/SOLAMI/datasets/multimodal_gen/data_gen/configs/api.yaml")

    if params.llm_settings.api_type in api_info.API_info.keys():
        params.llm_settings.update(api_info.API_info[params.llm_settings.api_type])
    if params.text_embedding_settings.api_type in api_info.API_info.keys():
        params.text_embedding_settings.update(api_info.API_info[params.text_embedding_settings.api_type])

    if args.output_dir is not None:
        params['save_settings']['save_path'] = args.output_path
    
    params['save_settings']['save_path'] = os.path.join(params['save_settings']['save_path'], f'{args.exper}')
    if not os.path.exists(params['save_settings']['save_path']):
        os.makedirs(params['save_settings']['save_path'])
            
    # output_path = os.path.join(params['save_settings']['save_path'], f'{args.exper}_peroid{args.period}_part{args.part}.{params["save_settings"]["save_format"]}')
    logger_path = os.path.join(os.path.dirname(params['save_settings']['save_path']), f'{args.exper}_peroid{args.period}_part{args.part}.log')
    logger = get_logger(save_path=logger_path, **params['logging_settings'])
    
    datasets = get_datasets(params['dataset_names'], data_configs=params['datasets'], logger=logger)
    
    ## get profile_data for generation
    profile_data = []
    if args.profile_data_path is not None:
        data_csv = pd.read_csv(args.profile_data_path)
        data_csv.fillna(-1, inplace=True)
        for index, row in data_csv.iterrows():
            profile_data.append([row['Agent Settings'], row['modified_topic'], row['gen_type']])
    else:
        profile_data.append(['', '', 'common'])
        
    params.pop('datasets')
    
    mappings = {
        'motion imitation': 'mi',
        'instruction following': 'if',
        'motion understanding': 'mu',
        'common': 'c',
        'script completion': 'sc',
        'agent conversation': 'ac',
    }
    
    
    for idx in tqdm(range(args.part + shift, len(profile_data), args.period)):
        content_type = profile_data[idx][2]
        if content_type not in ['common', 'motion imitation', 'instruction following', 'motion understanding']:
            logger.info('Invalid profile data at {}-th with topic {}'.format(str(idx), profile[idx][1]))
            continue
        if content_type == 'common':
        # for content_type in ['common']:
            methods = ['script completion', 'agent conversation']
        else:
            methods = ['script completion', ] #[random.choice(['script completion', 'agent conversation'])]
        for method in methods:
            params_tmp = params.copy()
            params_tmp['content_type'] = content_type
            params_tmp['conversation_settings']['method'] = method
            if content_type != 'common':
                REPEATS = 1
            else:
                REPEATS = params['conversation_settings']['NUM_EXAMPLES']
            for repeat in range(REPEATS):
                data_id = f"{idx}_{repeat}_{mappings[content_type]}_{mappings[method]}"
                output_path = os.path.join(params['save_settings']['save_path'], f'{args.exper}__{data_id}.{params["save_settings"]["save_format"]}')
                if os.path.exists(output_path):
                    continue
                profile = profile_data[idx]
                data_generated = generate_scripts(
                    profile=profile,
                    datasets=datasets,
                    tokens_blog=TOKENS_BLOG,
                    logger=logger,
                    **params_tmp,
                )
                if data_generated == {}:
                        continue
                data_generated['repeat'] = repeat
                data_generated['data_id'] = data_id
                save_res(data_generated, output_path, logger=logger)
                    
    ### Final
    cost_all, costs = calculate_api_cost(TOKENS_BLOG)
    logger.critical('Total cost ($): {}'.format(str(cost_all)))
    logger.critical('Detail costs ($): ')
    logger.critical(json.dumps(costs, indent=4))
    pass