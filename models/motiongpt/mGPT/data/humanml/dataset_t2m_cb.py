import rich
import random
import pickle
import os
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
import copy

class Text2MotionDatasetCB(data.Dataset):
    def __init__(
        self,
        data_root,
        mean,
        std,
        split,
        transform_mean,
        transform_std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text=False,
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length
        self.split = split
        # Data mean and std
        self.mean = mean
        self.std = std
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        # Data path
        split = 'train'
        self.cfg = copy.deepcopy(kwargs['cfg'])
        
        # split_file = pjoin(data_root, split + '.txt')
        # motion_dir = pjoin(data_root, code_path)
        # text_dir = pjoin(data_root, 'texts')
        
        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain':
            instructions = pjoin('/root/pengyang/codebase/SOLAMI/models/motiongpt/prepare/instructions', 'template_pretrain_new_simple.json')
        elif stage in ['lm_instruct', "lm_rl"]:
            instructions = pjoin('/root/pengyang/codebase/SOLAMI/models/motiongpt/prepare/instructions', 'template_instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")

        data_root = "SOLAMI_data/tmp_data"
        motion_repre = self.cfg['EXPER']['motion_repre']
        motion_part = self.cfg['EXPER']['motion_part']
        origin_token_dir = os.path.join("SOLAMI_data/tmp_data/tmp_tokens", motion_repre + '_' + motion_part)
        configs = {
            'dlp': {
                'text_embeddings': "SOLAMI_data/DLP-MoCap/embeddings/embeddings.npz",
                'dataset_items': "SOLAMI_data/DLP-MoCap/dataset_items.json",
                'dataset_root_dir': "SOLAMI_data/DLP-MoCap",
            },
            'humanml3d': {
                'text_embeddings': "SOLAMI_data/HumanML3D/embeddings/embeddings.npz",
                'dataset_items': "SOLAMI_data/HumanML3D/dataset_items.json",
                'dataset_root_dir': "SOLAMI_data/HumanML3D",
            },
            'interx': {
                'text_embeddings': "SOLAMI_data/Inter-X/embeddings/embeddings.npz",
                'dataset_items': "SOLAMI_data/Inter-X/dataset_items.json",
                'dataset_root_dir': "SOLAMI_data/Inter-X",
            },
        }
           
        dataset_items = {}     
        for key in configs:
            with open(configs[key]['dataset_items'], 'r') as f:
                dataset_items.update(json.load(f))
        print(f'Original Loaded total {len(dataset_items)} dataset items from all datasets')
        
        self.id_list = []
        if split == 'train':
            for key in dataset_items.keys():
                if dataset_items[key]['split_in_original'] == 'train':
                    self.id_list.append(key)
        elif split == 'val':
            for key in dataset_items.keys():
                if dataset_items[key]['split_in_original'] == 'val':
                    self.id_list.append(key)
        elif split == 'test':
            for key in dataset_items.keys():
                if dataset_items[key]['split_in_original'] == 'test':
                    self.id_list.append(key)

        if tiny or debug:
            # TODO Test inter mode
            if split == 'train':
                start_id = 25000
            else:
                start_id = 5000
            # self.id_list = self.id_list[start_id:]
            enumerator = enumerate(self.id_list[start_id:])
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading HumanML3D {split}",
                ))
            maxdata = 1e10
            subset = ''

        new_name_list = []
        length_list = []
        data_dict = {}

        
        if os.path.exists(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_repre}_{motion_part}_data.pkl')):
            if tiny or debug:
                with open(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_repre}_{motion_part}_data.pkl'),
                          'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with rich.progress.open(
                        pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_repre}_{motion_part}_data.pkl'),
                        'rb',
                        description=f"Loading HumanML3D {split}") as file:
                    data_dict = pickle.load(file)
            with open(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_repre}_{motion_part}_index.pkl'),
                      'rb') as file:
                name_list = pickle.load(file)
            for name in new_name_list:
                length_list.append(data_dict[name]['length'])
        else:
            for idx, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                
                data_item = dataset_items[name]
                
                dataset = data_item['dataset']
                dataset_root = configs[dataset]['dataset_root_dir']
                motion_token_path = os.path.join(origin_token_dir, data_item['motion_name']+ '.npz')
                motion_data_path = os.path.join(dataset_root, data_item['motion_data_path'])
                motion_data = np.load(motion_data_path, allow_pickle=True)
                
                try:
                    motion_token_data = np.load(motion_token_path, allow_pickle=True)
                except:
                    continue
                body_tokens = motion_token_data['body']
                if 'hand' in motion_token_data:
                    hand_tokens = motion_token_data['hand']
                    motion = np.concatenate([body_tokens, hand_tokens], axis=0)
                else:
                    motion = body_tokens
                    
                # print('motion len: ' + str(len(motion_data['smplx_feature'].item()['cont6d_local'].shape[0])) \
                #        + '  token len: ' + str(len(motion.shape[1])))
                
                # motion_data = dict(motion_data)
                
                # motion = motion_data['motion']
                start_frame = data_item['start_frame']
                end_frame = data_item['end_frame']
                
                if self.cfg['EXPER']['transform'] == True and (start_frame != 0 or end_frame != -1):
                    continue
                
                # motion = self.get_motion_feature(motion_data)
                
                # motion = motion[start_frame:end_frame]
                
                transform = None
                if self.cfg['EXPER']['transform'] == True:
                    transforms = motion_data['transforms'].item()
                    if data_item['dataset'] == 'interx':
                        if self.cfg['EXPER']['motion_repre'] == 'ske':
                            transform = np.concatenate([transforms['ske_relative_cont6d'], transforms['ske_relative_pos']], axis=0)
                        else:
                            transform = np.concatenate([transforms['smplx_relative_cont6d'], transforms['smplx_relative_pos']], axis=0)
                        transform = transform[[0, 2, 6, 7, 8]]
                
                # if (len(motion[0])) < self.min_motion_length or (len(motion[0])>= 300):
                #     continue

                # Read text
                text_data = []
                for i in range(len(data_item['text'])):
                    text_dict = {}
                    text_dict['caption'] = data_item['text'][i]
                    if i < len(data_item['tokens']):
                        tokens = data_item['tokens'][i].split(' ')
                        text_dict['tokens'] = tokens
                    else:
                        text_dict['tokens'] = None
                    text_data.append(text_dict)
                
                data_dict[name] = {
                        'motion': motion,
                        "length": len(motion[0]),
                        'text': text_data,
                        'transform': transform,
                        'partner_motion': data_item['next_partner_motion_name'],
                    }
                new_name_list.append(name)
                length_list.append(len(motion[0]))
                # except:
                #     pass

            name_list, length_list = zip(
                *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

            if tmpFile:
                os.makedirs(pjoin(data_root, 'tmp_tokens'), exist_ok=True)
                with open(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_repre}_{motion_part}_data.pkl'),
                          'wb') as file:
                    pickle.dump(data_dict, file)
                with open(pjoin(data_root, f'tmp_tokens/{split}{subset}_{motion_repre}_{motion_part}_index.pkl'),
                          'wb') as file:
                    pickle.dump(name_list, file)

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        print('length of data dict: ', len(self.data_dict))
        # self.name_list = name_list[start_id:start_id + maxdata]
        if motion_repre == 'ske':
            if motion_part == 'body':
                self.nfeats = 263
            else:
                self.nfeats = 623
        elif motion_repre == 'local cont6d':
            if motion_part == 'body':
                self.nfeats = 135
            else:
                self.nfeats = 315
        else:
            raise ValueError('wrong motion repre and motion part') 

        self.nlp = spacy.load('en_core_web_sm')
        self.std_text = std_text
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

    def __len__(self):

        return int(len(self.name_list) * len(self.tasks))

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = item // len(self.name_list)

        data_item = self.data_dict[self.name_list[data_idx]]
        m_token_list, m_length, text_list, transform, partner_name = \
            data_item['motion'], data_item['length'], data_item['text'], data_item['transform'], data_item['partner_motion']
         ### transform
        if partner_name is None or transform is None:
            # single person transform
            transform = np.random.normal(loc=self.transform_mean, scale=self.transform_std, size=self.transform_mean.shape)
            transform = (transform - self.transform_mean) / self.transform_std
        else:
            # two person transform
            transform = (transform - self.transform_mean) / self.transform_std
        
        # m_token_list, text_list = data['m_token_list'], data['text']

        partner_motion = None
        if partner_name is not None:
            if partner_name in self.data_dict:
                partner_motion = self.data_dict[partner_name]['motion']

        # m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN'
                        or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
                
            caption = ' '.join(word_list)
        
        # all_captions = [
        #     ' '.join([token.split('/')[0] for token in text_dic['tokens']])
        #     for text_dic in text_list
        # ]
        all_captions = [i['caption'] for i in text_list]
        if len(all_captions) >= 3:
            all_captions = all_captions[:3]
        else:
            for _ in range(3-len(all_captions)):
                all_captions.append(all_captions[0])
        
        coin = np.random.choice([False, False, True])

        m_tokens = m_token_list

        m_tokens_len = m_tokens.shape[-1]

        # if partner motion is none, randomly select a task from the other task
        tasks = self.tasks[task_idx]
        if partner_motion is None and tasks['class'] == 'interactive':
            while tasks['class'] == 'interactive':
                task_idx = np.random.randint(len(self.tasks))
                tasks = self.tasks[task_idx]
            

        return caption, m_tokens, m_tokens_len,  all_captions, transform, partner_motion, tasks
