# Adapted from MotionStreamer: humanml3d_272/dataset_eval_t2m.py
# Upstream repo: https://github.com/zju3dv/MotionStreamer/
# Source file: https://github.com/zju3dv/MotionStreamer/blob/main/humanml3d_272/dataset_eval_t2m.py

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)


class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, max_text_len = 20, unit_length = 4, split_file = None, skip_subsegments = False, return_motion_id = False, babel_filter = False):
        """
        Args:
            dataset_name: Name of dataset (e.g., 't2m_272')
            is_test: Whether this is test set
            max_text_len: Maximum text length
            unit_length: Unit length for motion cropping
            split_file: Optional custom split file (e.g., 'test_and_then.txt')
            skip_subsegments: If True, only use full-motion entries (skip sub-segment clips).
                             Useful for "and then" evaluation where we only want full motions.
            return_motion_id: If True, return motion_id along with (caption, motion, length).
            babel_filter: If True, only include motions that have BABEL annotations.
        """
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.skip_subsegments = skip_subsegments
        self.return_motion_id = return_motion_id
        self.babel_filter = babel_filter
        

        if dataset_name == 't2m_272':
            self.data_root = 'datasets/motions/humanml3d_272'
            self.motion_dir = pjoin(self.data_root, 'motion_data')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 300
            fps = 30
            self.meta_dir = 'datasets/motions/humanml3d_272/mean_std'
            if split_file is None:
                if is_test:
                    split_file = pjoin(self.data_root, 'split', 'test.txt')
                else:
                    split_file = pjoin(self.data_root, 'split', 'val.txt')
            else:
                # If split_file is provided, use it directly (can be absolute or relative path)
                if not os.path.isabs(split_file):
                    split_file = pjoin(self.data_root, 'split', split_file)
        print(f"Loading mean / std from: {os.path.abspath(pjoin(self.meta_dir, 'Mean.npy'))}")

        mean = np.load(pjoin(self.meta_dir, 'Mean.npy')) 
        mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'Std.npy'))

        min_motion_len = 60  # 30 fps

        data_dict = {}
        id_list = []

        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []


        for name in tqdm(id_list):
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
            if (len(motion)) < min_motion_len or (len(motion) >= self.max_motion_length):
                continue

            text_data = []
            flag = False
            with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                        
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    elif not self.skip_subsegments:
                        # Create sub-segment entries only if skip_subsegments is False
                        n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]           
                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= self.max_motion_length):
                            continue
                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        while new_name in data_dict:
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                        new_name_list.append(new_name)
                        length_list.append(len(n_motion))
                   

            if flag:
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data}
                new_name_list.append(name)
                length_list.append(len(motion))


        # Apply BABEL filtering if requested
        if self.babel_filter:
            new_name_list, length_list = self._filter_babel_motions(new_name_list, length_list, data_dict)
        
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def _filter_babel_motions(self, name_list, length_list, data_dict):
        """
        Filter motions to only include those that have BABEL annotations.
        
        Uses the streamer272_to_babel_mapping.json file which maps motion IDs to BABEL IDs.
        Only includes motions where 'matched' is True in the mapping.
        
        Args:
            name_list: List of motion names
            length_list: List of motion lengths
            data_dict: Dictionary of motion data
            
        Returns:
            Tuple of (filtered_name_list, filtered_length_list)
        """
        import json
        
        # Path to streamer272_to_babel_mapping.json (in babel-teach annotations folder)
        # Try multiple possible locations
        possible_paths = [
            os.path.join('datasets', 'annotations', 'babel-teach', 'streamer272_to_babel_mapping.json'),
            os.path.join(self.data_root, '..', '..', 'annotations', 'babel-teach', 'streamer272_to_babel_mapping.json'),
            os.path.join(self.data_root, 'streamer272_to_babel_mapping.json'),
        ]
        
        mapping_path = None
        for path in possible_paths:
            if os.path.exists(path):
                mapping_path = path
                break
        
        if mapping_path is None:
            print(f"WARNING: BABEL mapping file not found in any of: {possible_paths}")
            print("Cannot filter for BABEL annotations. Returning all motions.")
            return name_list, length_list
        
        print(f"Loading BABEL mapping from: {mapping_path}")
        with open(mapping_path, 'r') as f:
            babel_mapping = json.load(f)
        
        # Get set of motion IDs that have BABEL annotations (matched=True)
        babel_motion_ids = set(
            motion_id for motion_id, entry in babel_mapping.items() 
            if entry.get('matched', False) is True
        )
        print(f"BABEL mapping contains {len(babel_mapping)} total entries, {len(babel_motion_ids)} with matched=True")
        
        # Filter name_list and length_list
        filtered_names = []
        filtered_lengths = []
        
        for name, length in zip(name_list, length_list):
            # Handle sub-segment names (e.g., 'A_000123' -> '000123')
            base_name = name.split('_', 1)[-1] if '_' in name and name[0].isalpha() else name
            
            if base_name in babel_motion_ids:
                filtered_names.append(name)
                filtered_lengths.append(length)
        
        print(f"BABEL filtering: {len(name_list)} -> {len(filtered_names)} motions with BABEL annotations")
        
        if len(filtered_names) == 0:
            print("WARNING: No motions found with BABEL annotations!")
            print("Returning original list to avoid empty dataset.")
            return name_list, length_list
        
        return filtered_names, filtered_lengths

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # For test sets, prefer captions with "and then" if available, otherwise use first
        # For training/validation, use random selection
        if self.is_test:
            # Look for a caption with "and then"
            and_then_captions = [t for t in text_list if "and then" in t['caption'].lower()]
            if and_then_captions:
                text_data = and_then_captions[0]  # Use first caption with "and then"
            else:
                text_data = text_list[0]  # Fallback to first caption
        else:
            text_data = random.choice(text_list)
        caption = text_data['caption']

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        #"Motion Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        if self.return_motion_id:
            # Return motion_id (the base name, without sub-segment prefix)
            base_name = name.split('_', 1)[-1] if '_' in name and name[0].isalpha() else name
            return caption, motion, m_length, base_name
        
        return caption, motion, m_length




def collate_fn_with_motion_id(batch):
    """Collate function that handles tuples with motion_id."""
    batch.sort(key=lambda x: x[2], reverse=True)
    # Separate the motion_ids from the rest
    captions = [item[0] for item in batch]
    motions = torch.stack([torch.from_numpy(item[1]) for item in batch])
    lengths = torch.tensor([item[2] for item in batch])
    motion_ids = [item[3] for item in batch]
    return captions, motions, lengths, motion_ids


def DATALoader(dataset_name, is_test,
                batch_size,
                num_workers = 64, unit_length = 4, drop_last=True, split_file=None, skip_subsegments=False, 
                return_motion_id=False, babel_filter=False) : 
    
    dataset = Text2MotionDataset(
        dataset_name, is_test, 
        unit_length=unit_length, 
        split_file=split_file, 
        skip_subsegments=skip_subsegments,
        return_motion_id=return_motion_id,
        babel_filter=babel_filter
    )
    
    # Use appropriate collate function based on return_motion_id
    if return_motion_id:
        collate = collate_fn_with_motion_id
    else:
        collate = collate_fn
    
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle = True,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last = drop_last
    )
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
