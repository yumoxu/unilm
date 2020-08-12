# -*- coding: utf-8 -*-
import sys
from os.path import isfile, isdir, join
from os import listdir
from pathlib import Path

from qfs.query_tools import get_cid2masked_query

import io
import json
import numpy as np
from tqdm import tqdm

"""
    This module contains functions for loading queries, e,g., features and tesnors.

"""

class QueryFeatures(object):
    """
        Borrowed and revised from InputFeatures.
        A single set of features of data.
    """

    def __init__(self, summ_id, summ_mask, summ_seg_id, slot_id, slot_mask):
        self.summ_id = summ_id
        self.summ_mask = summ_mask
        self.summ_seg_id = summ_seg_id
        self.slot_id = slot_id
        self.slot_mask = slot_mask


def convert_summ_text_to_features(text, 
        max_seq_length,
        tokenizer, 
        cls_token='[CLS]',
        sep_token='[SEP]',
        mask_token='[MASK]',
        pad_token=0,
        pad_token_segment_id=0,
        max_num_slot=32,
        slot_as_cls=None,
        add_cls_at_begin=None,
        interval_segment=None):
    """
        Borrowed.
    """
    if type(text) is str:
        tokens = tokenizer.tokenize(text)
    elif type(text) is list:
        # organized via sentence; you can add customized ops here for each sentence
        # e.g., append [SEP] at each sentence's end
        import itertools
        tokens = [tokenizer.tokenize(sent) for sent in text]
        tokens = list(itertools.chain(*tokens))

    if slot_as_cls:  # add [CLS] before each [MASK], and [CLS] represents the slot
        new_tokens = []
        for token in tokens:
            if token == mask_token:
                new_tokens.append(cls_token)
            new_tokens.append(token)
        tokens = new_tokens
        slot_token = cls_token
    else:  # [MASK] represents the slot
        slot_token = mask_token
    
    # finalize tokens
    if add_cls_at_begin:
        tokens = [cls_token] + tokens[:max_seq_length-2] + [sep_token]
    else:
        tokens = tokens[:max_seq_length-1] + [sep_token]
    
    # finalize slots
    slot_ids = [i for i, t in enumerate(tokens) if t == slot_token][:max_num_slot]
    slot_mask = [1] * len(slot_ids)
    slot_padding_length = max_num_slot - len(slot_ids)
    slot_ids = slot_ids + ([pad_token] * slot_padding_length)
    slot_mask = slot_mask + ([0] * slot_padding_length)
    
    if interval_segment:
        _segs = [-1] + [i for i, t in enumerate(tokens) if t == sep_token]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segment_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segment_ids += s * [0]
            else:
                segment_ids += s * [1]
    else:
        segment_ids = [0] * len(tokens)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features_params = {
        'summ_id': input_ids,
        'summ_mask': input_mask,
        'summ_seg_id': segment_ids,
        'slot_id': slot_id, 
        'slot_mask': slot_mask,
    }
    return QueryFeatures(**features_params)


def features2batch(queries, 
        max_summ_seq_len,
        max_num_slot,
        slot_as_cls,
        add_cls_at_begin,
        interval_segment,
        tokenizer,
        pad_token=0,
        pad_token_segment_id=0,
        transform=None,
    ):
    """
        Borrowed and revised from func:feature2array
    """
    n_samples = len(queries)

    convert_params = {
        'max_summ_seq_len': max_summ_seq_len,
        'max_num_slot': max_num_slot,
        'slot_as_cls': slot_as_cls,
        'add_cls_at_begin': add_cls_at_begin,
        'interval_segment': interval_segment,
        'tokenizer': tokenizer,
        'pad_token': pad_token,
        'pad_token_segment_id': pad_token_segment_id,
    }

    features = []
    for query in queries:
        convert_params['query'] = query
        feature = convert_summ_text_to_features(**convert_params)
        features.append(feature)

    all_summ_id = np.zeros(shape=[n_samples, max_summ_seq_len], dtype=np.int64)
    all_summ_mask = np.zeros(shape=[n_samples, max_summ_seq_len], dtype=np.int64)
    all_summ_seg_id = np.zeros(shape=[n_samples, max_summ_seq_len], dtype=np.int64)
    all_slot_id = np.zeros(shape=[n_samples, max_num_slot], dtype=np.int64)
    all_slot_mask = np.zeros(shape=[n_samples, max_num_slot], dtype=np.int64)
    

    for i, f in enumerate(tqdm(features)):
        all_summ_id[i] = f.summ_id
        all_summ_mask[i] = f.summ_mask
        all_summ_seg_id[i] = f.summ_seg_id
        all_slot_id[i] = f.slot_id
        all_slot_mask[i] = f.slot_mask

    batch = {
        'summ_id': all_summ_id,
        'summ_seg_id': all_summ_seg_id,
        'summ_mask': all_summ_mask,
        'slot_id': all_slot_id,
        'slot_mask': all_slot_mask,
    }

    # return all_summ_id, all_summ_mask, all_summ_seg_id, all_slot_id, all_slot_mask
    if transform:
        batch = transform(batch)
    return batch


def get_test_cc_ids(year):
    proj_root = Path('/disk/nfs/ostrom/s1617290/shiftsum')
    dp_data = proj_root / 'data'
    dp_duc_cluster = dp_data / 'duc_cluster' / year
    SEP = '_'

    all_cc_ids = [SEP.join((year, fn)) for fn in listdir(dp_duc_cluster)
        if isdir(join(dp_duc_cluster, fn))]
    return all_cc_ids


class ToTensor(object):
    """
        Borrowed from shiftsum/src/data/data_tools.py.

        Convert ndarrays in sample to Tensors.
    """

    def __call__(self, numpy_dict):
        for (k, v) in numpy_dict.items():
            if k.endswith('_ids'):
                v = v.type(torch.LongTensor)  # for embedding look up
                v = v.cuda()
            numpy_dict[k] = v
        return numpy_dict


def get_query_tensors(start_idx, 
        end_idx, 
        year,
        query_type, 
        max_summ_seq_len, 
        max_num_slot, 
        slot_as_cls, add_cls_at_begin, interval_segment,
        tokenizer):
    """
        Given cid indices (determined via start_idx and end_idx), get query-related tensors.

    """
    assert query_type
    query_dict = get_cid2masked_query(query_type)
    cids = get_test_cc_ids(year)[start_idx:end_idx]
    queries =  [query_dict[cid] for cid in cids]

    base_params = {
        'queries': queries,
        'max_summ_seq_len': max_summ_seq_len,
        'max_num_slot': max_num_slot,
        'slot_as_cls': slot_as_cls,
        'add_cls_at_begin': add_cls_at_begin,
        'interval_segment': interval_segment,
        'tokenizer': tokenizer,
        'pad_token': 0,
        'pad_token_segment_id': 0,
        'transform': ToTensor()
    }

    batch = features2batch(**base_params)
    return batch
