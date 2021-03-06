# coding=utf-8
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from operator import add
from typing import List, Optional, Tuple, Union
from torch.autograd import Variable
from tqdm import trange
from transformers.file_utils import cached_path

from torch.nn.modules.loss import _Loss


class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
    'unilm-base-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin",
    'unilm-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin",
    'unilm1-base-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin",
    'unilm1-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin",
    'unilm1.2-base-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased.bin"
}
CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 new_pos_ids=False,
                 initializer_range=0.02,
                 task_idx=None,
                 fp32_embedding=False,
                 ffn_type=0,
                 label_smoothing=None,
                 num_qkv=0,
                 seg_emb=False,
                 source_type_id=0, 
                 target_type_id=1,
                 no_segment_embedding=False, **kwargs):
        """Constructs BertConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.new_pos_ids = new_pos_ids
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.ffn_type = ffn_type
            self.label_smoothing = label_smoothing
            self.num_qkv = num_qkv
            self.seg_emb = seg_emb
            self.no_segment_embedding = no_segment_embedding
            self.source_type_id = source_type_id
            self.target_type_id = target_type_id
            if type_vocab_size == 0:
                self.no_segment_embedding = True
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        if config.no_segment_embedding:
            self.token_type_embeddings = None
        else:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        if hasattr(config, 'new_pos_ids') and config.new_pos_ids:
            self.num_pos_emb = 4
        else:
            self.num_pos_emb = 1
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size * self.num_pos_emb)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, task_idx=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if self.num_pos_emb > 1:
            num_batch = position_embeddings.size(0)
            num_pos = position_embeddings.size(1)
            position_embeddings = position_embeddings.view(
                num_batch, num_pos, self.num_pos_emb, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]

        embeddings = words_embeddings + position_embeddings

        if self.token_type_embeddings is not None:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if hasattr(config, 'num_qkv') and (config.num_qkv > 1):
            self.num_qkv = config.num_qkv
        else:
            self.num_qkv = 1

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(config.hidden_size,
                             self.all_head_size * self.num_qkv)
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size * self.num_qkv)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.uni_debug_flag = True if os.getenv(
            'UNI_DEBUG_FLAG', '') else False
        if self.uni_debug_flag:
            self.register_buffer('debug_attention_probs',
                                 torch.zeros((512, 512)))
        if hasattr(config, 'seg_emb') and config.seg_emb:
            self.b_q_s = nn.Parameter(torch.zeros(
                1, self.num_attention_heads, 1, self.attention_head_size))
            self.seg_emb = nn.Embedding(
                config.type_vocab_size, self.all_head_size)
        else:
            self.b_q_s = None
            self.seg_emb = None

    def transpose_for_scores(self, x, mask_qkv=None):
        if self.num_qkv > 1:
            sz = x.size()[:-1] + (self.num_qkv,
                                  self.num_attention_heads, self.all_head_size)
            # (batch, pos, num_qkv, head, head_hid)
            x = x.view(*sz)
            if mask_qkv is None:
                x = x[:, :, 0, :, :]
            elif isinstance(mask_qkv, int):
                x = x[:, :, mask_qkv, :, :]
            else:
                # mask_qkv: (batch, pos)
                if mask_qkv.size(1) > sz[1]:
                    mask_qkv = mask_qkv[:, :sz[1]]
                # -> x: (batch, pos, head, head_hid)
                x = x.gather(2, mask_qkv.view(sz[0], sz[1], 1, 1, 1).expand(
                    sz[0], sz[1], 1, sz[3], sz[4])).squeeze(2)
        else:
            sz = x.size()[:-1] + (self.num_attention_heads,
                                  self.attention_head_size)
            # (batch, pos, head, head_hid)
            x = x.view(*sz)
        # (batch, head, pos, head_hid)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None,
                mask_qkv=None, seg_ids=None, key_history=None, value_history=None,
                key_cache=None, value_cache=None,
                ):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = F.linear(hidden_states, self.key.weight)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = F.linear(x_states, self.key.weight)
            mixed_value_layer = self.value(x_states)

        if key_cache is not None and isinstance(key_cache, list):
            key_cache.append(mixed_key_layer)
            mixed_key_layer = torch.cat(key_cache, dim=1)

        if value_cache is not None and isinstance(value_cache, list):
            value_cache.append(mixed_value_layer)
            mixed_value_layer = torch.cat(value_cache, dim=1)

        query_layer = self.transpose_for_scores(mixed_query_layer, mask_qkv)
        key_layer = self.transpose_for_scores(mixed_key_layer, mask_qkv)
        value_layer = self.transpose_for_scores(mixed_value_layer, mask_qkv)

        if key_history is not None and not isinstance(key_history, list):
            key_layer = torch.cat((key_history, key_layer), dim=-2)
            value_layer = torch.cat((value_history, value_layer), dim=-2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch, head, pos, pos)
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        # print('--------Self Attention--------')
        # print(f'query_layer: {query_layer.size()}')
        # print(f'key_layer: {key_layer.size()}')
        # print(f'value_layer: {value_layer.size()}')
        # print(f'attention_scores: {attention_scores.size()}')
        # print(f'attention_mask: {attention_mask.size()}')

        if self.seg_emb is not None:
            seg_rep = self.seg_emb(seg_ids)
            # (batch, pos, head, head_hid)
            seg_rep = seg_rep.view(seg_rep.size(0), seg_rep.size(
                1), self.num_attention_heads, self.attention_head_size)
            qs = torch.einsum('bnih,bjnh->bnij',
                              query_layer + self.b_q_s, seg_rep)
            attention_scores = attention_scores + qs

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.uni_debug_flag:
            _pos = attention_probs.size(-1)
            self.debug_attention_probs[:_pos, :_pos].copy_(
                attention_probs[0].mean(0).view(_pos, _pos))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if isinstance(key_history, list):
            key_history.append(key_layer)
        if isinstance(value_history, list):
            value_history.append(value_layer)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None,
                mask_qkv=None, seg_ids=None, key_history=None, value_history=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states,
            mask_qkv=mask_qkv, seg_ids=seg_ids, key_history=key_history, value_history=value_history)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerFFN(nn.Module):
    def __init__(self, config):
        super(TransformerFFN, self).__init__()
        self.ffn_type = config.ffn_type
        assert self.ffn_type in (1, 2)
        if self.ffn_type in (1, 2):
            self.wx0 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (2,):
            self.wx1 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (1, 2):
            self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        if self.ffn_type in (1, 2):
            x0 = self.wx0(x)
            if self.ffn_type == 1:
                x1 = x
            elif self.ffn_type == 2:
                x1 = self.wx1(x)
            out = self.output(x0 * x1)
        out = self.dropout(out)
        out = self.LayerNorm(out + x)
        return out


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.ffn_type = config.ffn_type
        if self.ffn_type:
            self.ffn = TransformerFFN(config)
        else:
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None,
                mask_qkv=None, seg_ids=None, key_history=None, value_history=None):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states,
            mask_qkv=mask_qkv, seg_ids=seg_ids, key_history=key_history, value_history=value_history)
        if self.ffn_type:
            layer_output = self.ffn(attention_output)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, prev_embedding=None,
                prev_encoded_layers=None, mask_qkv=None, seg_ids=None, key_history=None, value_history=None):
        # history embedding and encoded layer must be simultanously given
        assert (prev_embedding is None) == (prev_encoded_layers is None)

        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for i, layer_module in enumerate(self.layer):
                set_key = None
                if isinstance(key_history, list):
                    set_key = key_history if len(key_history) < len(self.layer) else key_history[i]
                set_value = None
                if isinstance(value_history, list):
                    set_value = value_history if len(key_history) < len(self.layer) else value_history[i]
                hidden_states = layer_module(
                    hidden_states, attention_mask, mask_qkv=mask_qkv, seg_ids=seg_ids,
                    key_history=set_key, value_history=set_value)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor

        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) -> (batch, num_pos, hid)
            hidden_states = hidden_states.view(
                num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if self.fp32_embedding:
            hidden_states = F.linear(self.type_converter(hidden_states), self.type_converter(
                self.decoder.weight), self.type_converter(self.bias))
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output, task_idx=None):
        prediction_scores = self.predictions(sequence_output, task_idx)
        if pooled_output is None:
            seq_relationship_score = None
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # module.weight.data.copy_(torch.Tensor(
            #     truncnorm.rvs(-1, 1, size=list(module.weight.data.shape)) * self.config.initializer_range))
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, config, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        logger.info("Model config {}".format(config))

        # clean the arguments in kwargs
        for arg_clean in ('config_path', 'type_vocab_size', 'relax_projection', 'new_pos_ids', 'task_idx',
                          'max_position_embeddings', 'fp32_embedding', 'ffn_type', 'label_smoothing',
                          'hidden_dropout_prob', 'attention_probs_dropout_prob', 'num_qkv', 'seg_emb',
                          'word_emb_map', 'num_labels', 'num_rel', 'num_sentlvl_labels'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(pretrained_model_name, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def rescale_some_parameters(self):
        for layer_id, layer in enumerate(self.encoder.layer):
            layer.attention.output.dense.weight.data.div_(
                math.sqrt(2.0 * (layer_id + 1)))
            layer.output.dense.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                mask_qkv=None, task_idx=None, key_history=None, value_history=None, position_ids=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, task_idx=task_idx, position_ids=position_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      mask_qkv=mask_qkv, seg_ids=token_type_ids,
                                      key_history=key_history, value_history=value_history)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, output_all_encoded_layers=True,
                prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, task_idx=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids, task_idx=task_idx)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv,
                                      seg_ids=token_type_ids)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class MargeDiscriminator(nn.Module):
    def __init__(self, bert_model, label, loss_idx, pool_func='ls', device='cuda', hidden_size=768, eps=1e-7, fp16=False):
        super(MargeDiscriminator, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.pool_func = pool_func
        self.eps = eps
        self.label = label
        self.loss_idx = loss_idx
        self.device = device
        self.fp16 = fp16

        # attributes that should be updated per batch
        self.slot_rep = None
        self.slot_mask = None

    def _match(self, cand_rep, slot_rep, instc_mask):
        """

        :param cand_rep: d_batch * d_embed
        :param slot_rep: d_batch * max_ns * d_embed
        :param instc_mask: d_batch * max_ns

        :return:
            score: d_batch * max_ns
        """
       
        cand_rep = torch.unsqueeze(cand_rep, dim=-1)  # d_batch * d_embed * 1

        K = int(cand_rep.size(0) / slot_rep.size(0))
        if K > 1:
            d_embed = cand_rep.size(-2)
            d_batch = slot_rep.size(0)
            slot_rep = torch.unsqueeze(slot_rep, dim=1) # d_batch * 1 * max_ns * d_embed
            cand_rep = torch.reshape(cand_rep, [d_batch, K, d_embed, 1])  # d_batch * K * d_embed * 1
            # print(f'K: {K}')
            instc_score_in = torch.matmul(slot_rep, cand_rep)  # d_batch * K * max_ns * 1
            instc_score_in = torch.squeeze(instc_score_in, dim=-1)  # d_batch * K * max_ns
            instc_score_in = instc_score_in / np.sqrt(self.hidden_size)
            instc_score = torch.sigmoid(instc_score_in)  # d_batch * K * max_ns
            
            # mask
            instc_mask = torch.unsqueeze(instc_mask, dim=1)  # d_batch * 1 * max_ns
            instc_score = instc_score * instc_mask
            instc_score = torch.reshape(instc_score, (-1, instc_score.size(-1)))  # (d_batch * K) * max_ns
        else:
            instc_score_in = torch.matmul(slot_rep, cand_rep)  # d_batch * max_ns * 1
            instc_score_in = torch.squeeze(instc_score_in, dim=-1) / np.sqrt(self.hidden_size)  # d_batch * max_ns
            instc_score = torch.sigmoid(instc_score_in)
            instc_score = instc_score * instc_mask
        return instc_score
    
    def _pool(self, instc_score, instc_mask=None):
        """

        :param instc_score: d_batch * max_ns
        :param instc_mask: d_batch * max_ns
        :return:
            group_score: d_batch * 1
        """
        if self.pool_func == 'avg':
            nom = torch.sum(instc_score, dim=-1, keepdim=True)  # d_batch * 1
            n_instc = torch.sum(instc_mask, dim=-1, keepdims=True)  # d_batch * 1
            return nom / n_instc

        elif self.pool_func == 'max':
            instc_score.masked_fill((1-instc_mask).type(torch.uint8), float('-inf'))
            # instc_score[1-instc_mask] = float('-inf')
            return torch.max(instc_score, dim=-1)[0]

        elif self.pool_func == 'ls':
            nom = torch.sum(instc_score ** 2, dim=-1, keepdim=True)
            denom = torch.sum(instc_score, dim=-1, keepdim=True) + self.eps
            return nom / denom

        else:
            raise ValueError(f'Invalid pool_func: {self.pool_func}')

    def _adapt_var(self, var):
        if self.fp16:
            return var.half()
        else:
            return var.float()
    
    def get_loss(self, pred):
        pred = pred.view(-1)
        label = torch.tensor(pred.shape[0] * [self.label], device=self.device, dtype=torch.float)
        label = self._adapt_var(label)
        # print(f'label: {label.size()}, pred: {pred.size()}')
        loss = MSELoss()(pred, label)
        return loss
    
    def get_rand_slot_rep(self, d_batch, max_n_slot):
        """
            For debug.
        """
        slot_rep = torch.rand(size=(d_batch, max_n_slot, 768), device=self.device, dtype=torch.half)
        return slot_rep
    
    # def _forward(self, summ_id, summ_seg_id, summ_mask, slot_id, slot_mask, cand_rep):
    #     if self.new_batch or self.slot_rep is None:  # only update for new data
    #         self.get_slot_rep(summ_id, summ_seg_id, summ_mask, slot_id, slot_mask)
    
    def _forward_unit_test_0(self, cand_rep):
        """
            
            For unit test. Forward with random slot representations. 
        """
        slot_rep = self.get_rand_slot_rep(d_batch=cand_rep.size(0), max_n_slot=8)
        d_embed = cand_rep.size()[-1]
        cand_rep = torch.unsqueeze(cand_rep, dim=-1)  # d_batch * d_embed * 1
        instc_score_in = torch.matmul(slot_rep, cand_rep)  # d_batch * max_ns * 1
        instc_score_in = torch.squeeze(instc_score_in, dim=-1) / np.sqrt(self.hidden_size)  # d_batch * max_ns
        instc_score = torch.sigmoid(instc_score_in)

        # instc_score = self._match(cand_rep, slot_rep, instc_mask=slot_mask)
        group_score = self._pool(instc_score)  # d_batch * 1
        group_score = torch.clamp(group_score, min=self.eps, max=1-self.eps)  # in (0, 1)
        # print(f'group_score: {group_score[0]}\ninstc_score: {instc_score[0]}')

        if self.loss_idx >= 0:
            pred = instc_score[self.loss_idx]
            loss = self.get_loss(pred=instc_score[self.loss_idx])
        else:
            loss = self.get_loss(pred=group_score)

        return loss, group_score, instc_score

    def _forward_unit_test_1(self, cand_rep):
        """
            
            For unit test. Forward with random scores.

            cand_rep: d_batch * d_embed
        """
        cand_rep = torch.sigmoid(cand_rep)
        group_score = torch.max(cand_rep, dim=-1)[0]  # d_batch * 1
        group_score = torch.clamp(group_score, min=self.eps, max=1-self.eps)  # in (0, 1)
        loss = self.get_loss(pred=group_score)

        return loss, group_score, None
    
    def init_slot_rep(self, summ_id, summ_seg_id, summ_mask, slot_id, slot_mask):
        max_summ_seq_len = summ_id.size(1)
        
        # with torch.cuda.device(0):
        summ_rep = self.bert_model(summ_id, 
            token_type_ids=summ_seg_id, 
            attention_mask=summ_mask)[0].view(-1, max_summ_seq_len, self.hidden_size)

        # select class reps
        slot_rep = summ_rep[torch.arange(summ_rep.size(0)).unsqueeze(1), slot_id]
        self.slot_mask = self._adapt_var(slot_mask)
        self.slot_rep = slot_rep * self.slot_mask[:, :, None]
        self.slot_rep.detach()
    
    def forward(self, cand_rep):
        assert (self.slot_rep is not None) or (self.slot_mask is not None), \
            'Init self.slot_rep and self.slot_mask before calling self.foward()!'

        # with torch.cuda.device(0):
        instc_score = self._match(cand_rep, self.slot_rep, instc_mask=self.slot_mask)
        group_score = self._pool(instc_score, instc_mask=self.slot_mask)  # d_batch * 1
        group_score = torch.clamp(group_score, min=self.eps, max=1-self.eps)  # in (0, 1)

        if self.loss_idx >= 0:
            pred = instc_score[self.loss_idx]
            loss = self.get_loss(pred=instc_score[self.loss_idx])
        else:
            loss = self.get_loss(pred=group_score)

        return loss, group_score, instc_score


class BertEmbeddingsforQueryFocus(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddingsforQueryFocus, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        if config.no_segment_embedding:
            self.token_type_embeddings = None
        else:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        if hasattr(config, 'new_pos_ids') and config.new_pos_ids:
            self.num_pos_emb = 4
        else:
            self.num_pos_emb = 1
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size * self.num_pos_emb)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, words_embeddings=None, token_type_ids=None, position_ids=None, task_idx=None):
        """
             Revision: support pre-calculated words_embeddings.

        """
        if input_ids is not None and words_embeddings is not None:
            raise ValueError("You cannot specify both input_ids and words_embeddings at the same time")
        elif input_ids is not None:
            input_shape = list(input_ids.size())
            device = input_ids.device
            words_embeddings = self.word_embeddings(input_ids)
        elif words_embeddings is not None:
            input_shape = list(words_embeddings.size()[:-1])
            device = words_embeddings.device
        else:
            raise ValueError("You have to specify either input_ids or words_embeddings")

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape)

        position_embeddings = self.position_embeddings(position_ids)

        if self.num_pos_emb > 1:
            num_batch = position_embeddings.size(0)
            num_pos = position_embeddings.size(1)
            position_embeddings = position_embeddings.view(
                num_batch, num_pos, self.num_pos_emb, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]

        embeddings = words_embeddings + position_embeddings

        if self.token_type_embeddings is not None:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelIncrForQueryFocus(BertModel):
    def __init__(self, config):
        super(BertModelIncrForQueryFocus, self).__init__(config)
        self.embeddings = BertEmbeddingsforQueryFocus(config)

    def get_extended_attention_mask(self, input_shape, token_type_ids, attention_mask):
        """
            Revision: use input_shape as parameter, instead of input_ids, to support input_embeds.

        """
        if attention_mask is None:
            attention_mask = torch.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_shape)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, token_type_ids, position_ids, attention_mask, 
                input_ids=None, input_embeds=None, 
                output_all_encoded_layers=True,
                prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, task_idx=None):
        """
            Revision: support input_embeds.

            Revised items:
                (1) Overwrite self.get_extended_attention_mask() to remove the dependency of input_ids
                (2) Overwrite self.BertEmbeddingsforQueryFocus to support pre-calculted input_embeds

            Return:
                embedding_output and encoded_layers are for the input_ids.
                Usually after the first time encoding the context, input_ids are for only the current generation step.
        """
        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        elif input_ids is not None:
            input_shape = list(input_ids.size())
        elif input_embeds is not None:
            input_shape = list(input_embeds.size()[:-1])
        else:
            raise ValueError("You have to specify either input_ids or input_embeds")
        
        extended_attention_mask = self.get_extended_attention_mask(input_shape, token_type_ids, attention_mask)

        embedding_output = self.embeddings(input_ids=input_ids, words_embeddings=input_embeds, 
            token_type_ids=token_type_ids, position_ids=position_ids, task_idx=task_idx)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers, 
                                      mask_qkv=mask_qkv,
                                      seg_ids=token_type_ids)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


# Global vars for PP
# SMALL_CONST = 1e-15
SMALL_CONST = 1e-6
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
DEBUG = 4
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
    'debug': DEBUG,
}

class BertForQueryFocusedDecoder(PreTrainedBertModel):
    """refer to BertForPreTraining
    """
    def __init__(self, config, mask_word_id=0, num_labels=2, num_rel=0,
                 search_beam_size=1, length_penalty=1.0, eos_id=0, sos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, ngram_size=3, min_len=0, mode="s2s", pos_shift=False,
                 stepsize=0.01,
                 temperature=1.0,
                 num_iterations=3,
                 grad_length=10000,
                 horizon_length=1,
                 window_length=0,
                 decay=False,
                 gamma=1.5,
                 gm_scale=0.95,
                 kl_scale=0.01,
                 verbosity=REGULAR,
                 device='cuda',
                 fp16=False,
                 discriminator=None):
        super(BertForQueryFocusedDecoder, self).__init__(config)
        self.bert = BertModelIncrForQueryFocus(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.num_rel = num_rel
        if self.num_rel > 0:
            self.crit_pair_rel = BertPreTrainingPairRel(
                config, num_rel=num_rel)
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pos_shift = pos_shift

        # configs for pplm        
        self.verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

        self.stepsize = stepsize
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.grad_length = grad_length
        self.horizon_length = horizon_length
        self.window_length = window_length
        self.decay = decay
        self.gamma = gamma
        self.gm_scale = gm_scale
        self.kl_scale = kl_scale

        self.device = device
        self.fp16 = fp16

        self.discriminator = discriminator

    def to_var(self, x, requires_grad, volatile=False):
        if torch.cuda.is_available() and self.device == 'cuda':
            x = x.cuda()
        elif self.device != 'cuda':
            x = x.to(self.device)
        
        if self.fp16:
            x = x.half()

        return Variable(x, requires_grad=requires_grad, volatile=volatile)

    @torch.enable_grad()
    def perturb_past_backup(
            self,
            discriminator,
            input_shape,
            token_type_ids,
            position_ids,
            attention_mask,
            task_idx,
            mask_qkv,
            curr_ids,
            next_pos,
            stepsize,
            prev_embedding=None,
            prev_encoded_layers=None,
            new_embedding=None,
            new_encoded_layers=None,
            unpert_logits=None,
            accumulated_hidden=None,
            layer_grad_norms=None,
            embedding_grad_norm=None,
            mask_ids=None,
            sos_ids=None
    ):
        """
            This is an archived version that confuses the use of past and unpert_past.
        """
        def build_unpert_past():
            """
                In the first perturbation, new_* cover {context, [MASK]}, and prev_* are not provided.
                Therefore, we use new_*[:-1] as unpert_*.
                
                In the later perturbation, prev_* are provided, and we use them as unpert_*.
            """
            assert not self.pos_shift
            if prev_embedding is None:  # is the first, new_embedding is context + MASK; perturb the context
                unpert_embedding = new_embedding[:, :-1, :]
            else:
                # unpert_embedding = torch.cat((prev_embedding, new_embedding[:, :-2, :]), dim=1)
                unpert_embedding = prev_embedding
            
            if prev_encoded_layers is None:
                unpert_layers = [x[:, :-1, :] for x in new_encoded_layers]
            else:
                unpert_layers = prev_encoded_layers
            return unpert_embedding, unpert_layers

        unpert_embedding, unpert_layers = build_unpert_past()

        layer_grad_accumulator = [
            (np.zeros(p.shape).astype("float32"))
            for p in unpert_layers[:-1]
        ]
        embedding_grad_accumulator = np.zeros(unpert_embedding.shape).astype("float32")

        # if accumulated_hidden is None:
            # accumulated_hidden = 0
        accumulated_hidden = torch.sum(unpert_layers[-1], dim=1)  # sum of the current history
        # print(f'accumulated_hidden: {accumulated_hidden}')

        if self.decay:
            decay_mask = torch.arange(0., 1.0 + SMALL_CONST, 1.0 / (self.window_length))[1:]
        else:
            decay_mask = 1.0

        # Generate a mask is gradient perturbated is based on a past window
        curr_length = embedding_grad_accumulator.shape[-2]  # current history length (context + generation so far)
        if self.verbosity_level >= VERBOSE:
            print(f'curr_length: {curr_length}, embedding_grad_accumulator shape: {embedding_grad_accumulator.shape}')

        if curr_length > self.window_length and self.window_length > 0:
            d_batch, _, d_hidden = unpert_embedding.size()
            ones_key_val_shape = (d_batch, self.window_length, d_hidden)
            zeros_key_val_shape = (d_batch, curr_length - self.window_length, d_hidden)
            
            ones_mask = torch.ones(ones_key_val_shape)
            # ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
            ones_mask = decay_mask * ones_mask.permute(0, 2, 1)
            ones_mask = ones_mask.permute(0, 2, 1)

            window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(self.device)
        else:
            # window_mask = torch.ones_like(past[0]).to(device)
            window_mask = torch.ones_like(unpert_embedding).to(self.device)

        # accumulate perturbations for num_iterations
        loss_per_iter = []
        new_accumulated_hidden = None
        for i in range(self.num_iterations):
            if self.verbosity_level >= VERBOSE:
                print(f'Perturb Iter {next_pos}.{i + 1}')
            
            curr_layer_perturbation = [self.to_var(torch.from_numpy(p_), requires_grad=True)
                for p_ in layer_grad_accumulator]
            curr_embedding_perturbation = self.to_var(torch.from_numpy(embedding_grad_accumulator), requires_grad=True)

            # Compute hidden using perturbed past
            perturbed_layers = list(map(add, unpert_layers[:-1], curr_layer_perturbation))
            perturbed_layers.append(unpert_layers[-1])
            pertubed_embedding = unpert_embedding + curr_embedding_perturbation

            step_base_params = {
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'task_idx': task_idx,
                'mask_qkv': mask_qkv,
                'next_pos': next_pos,
                'mask_ids': mask_ids,
                'sos_ids': sos_ids,
            }
            logits, new_embedding, new_encoded_layers = self.step_for_current_perturb(**step_base_params,
                input_shape=input_shape,
                perturbed_embedding=pertubed_embedding, 
                perturbed_layers=perturbed_layers,
                curr_ids=curr_ids
            )

            # hidden = new_encoded_layers[-1] # last hidden layer, for only the current input
            # use only the last token for accumulation
            hidden = new_encoded_layers[-1][:, -1, :]  # last hidden state of the last hidden layer
            # print(f'hidden: {hidden.size()}')
            # new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
            new_accumulated_hidden = accumulated_hidden + hidden.detach()
            
            probs = F.softmax(logits, dim=-1)

            loss = 0.0
            # loss_list = []

            if self.horizon_length == 1:
                # perturb again for the future
                # as input, use the last expected embedding (intead of actual embedding of a token)
                # Is it the right not updating the history via including t+1?
                wte = self.bert.embeddings.word_embeddings  # n_vocab * n_hidden
                inputs_embeds = torch.matmul(probs, wte.weight.data)

                next_embedding, next_layers = self.step_for_future_perturb(**step_base_params,
                    input_shape=input_shape,
                    input_embeds=inputs_embeds,
                    prev_embedding=unpert_embedding, 
                    prev_encoded_layers=unpert_layers
                )
                # next_hidden = next_layers[-1]
                next_hidden = next_layers[-1][:, -1, :]  # last hidden state of the last hidden layer
                # print(f'next_hidden: {next_hidden.size()}')
                # new_accumulated_hidden = new_accumulated_hidden + torch.sum(next_hidden, dim=1)
                new_accumulated_hidden = new_accumulated_hidden + next_hidden
            
            elif self.horizon_length > 1:
                raise ValueError('Cannot set horizon_length over 1 since it has not been implemented.')

            # 1 is the perturbation for the present, horizon_length is for the future
            cand_rep = new_accumulated_hidden / (curr_length + 1 + self.horizon_length)
            discrim_loss, group_score, instc_score = discriminator(cand_rep)
            if self.verbosity_level >= VERY_VERBOSE:
                print(f'group_score: {group_score[0]}\ninstc_score: {instc_score[0]}')
                print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            # loss_list.append(discrim_loss)

            kl_loss = 0.0
            if self.kl_scale > 0.0:
                # unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
                unpert_probs = F.softmax(unpert_logits, dim=-1)
                # if self.fp16:
                #     unpert_correction = SMALL_CONST * (unpert_probs <= SMALL_CONST).half().to(self.device).detach()
                #     correction = SMALL_CONST * (probs <= SMALL_CONST).half().to(self.device).detach()
                # else:
                #     unpert_probs = SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(self.device).detach()
                #     correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(self.device).detach()

                # unpert_probs = (unpert_probs + unpert_correction.detach())
                # corrected_probs = probs + correction.detach()
                # print(f'corrected_probs: {corrected_probs}')
                
                # print(f'probs: {probs}')
                # print(f'unpert_probs: {unpert_probs}')
                # div = corrected_probs * (corrected_probs / unpert_probs).log()
                kl_loss_layer = torch.nn.KLDivLoss(reduction='sum')
                # div = kl_loss_layer(probs, unpert_probs)
                div = kl_loss_layer(logits, unpert_probs)
                kl_loss = self.kl_scale * div

                if self.verbosity_level >= VERY_VERBOSE:
                    print(f'kl_loss: {kl_loss.data.cpu().numpy()}')
                loss += kl_loss

            loss_per_iter.append(loss.data.cpu().numpy())
            if self.verbosity_level >= VERBOSE:
                print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

            if self.verbosity_level >= DEBUG:
                discrim_loss.retain_grad()
                group_score.retain_grad()
                cand_rep.retain_grad()
                new_accumulated_hidden.retain_grad()
                # print(f'curr_layer_perturbation 0: {curr_layer_perturbation[0].requires_grad}, {curr_layer_perturbation[0]}')
                # print(f'cand_rep: {cand_rep.requires_grad}, {cand_rep}')
                # print(f'group_score: {group_score.requires_grad}, {group_score}')
                # print(f'discrim_loss: {discrim_loss.requires_grad}, {discrim_loss}')
                # print(f'loss: {loss.requires_grad}, {loss}')

            # compute gradients
            loss.backward()
            # loss.backward(retain_graph=True)

            if self.verbosity_level >= DEBUG:
                print(f'Grad of discrim_loss: {discrim_loss.grad}')
                print(f'Grad of group_score: {group_score.grad}')
                print(f'Grad of cand_rep: {cand_rep.grad}')
                print(f'Grad of new_accumulated_hidden: {new_accumulated_hidden.grad}')
                for index, p_ in enumerate(curr_layer_perturbation):
                    print(f'Grad of curr_layer_perturbation[{index}]: {p_.grad}')

            # calculate gradient norms
            if layer_grad_norms is not None and embedding_grad_norm is not None:
                layer_grad_norms = [
                    torch.max(layer_grad_norms[index], torch.norm(p_.grad * window_mask))
                    for index, p_ in enumerate(curr_layer_perturbation)
                ]
                embedding_grad_norm = torch.max(embedding_grad_norm, torch.norm(curr_embedding_perturbation.grad * window_mask))
            else:
                layer_grad_norms = [
                    (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                    for index, p_ in enumerate(curr_layer_perturbation)
                ]
                embedding_grad_norm = (torch.norm(curr_embedding_perturbation.grad * window_mask) + SMALL_CONST)

            def finalize_grad(var, norm):
                return -stepsize * (var.grad * window_mask / norm ** self.gamma).data.cpu().numpy()

            layer_grad = [
                finalize_grad(p_, norm=layer_grad_norms[index])
                for index, p_ in enumerate(curr_layer_perturbation)
            ]
            embedding_grad = finalize_grad(curr_embedding_perturbation, norm=embedding_grad_norm)

            # accumulate gradient
            layer_grad_accumulator = list(map(add, layer_grad, layer_grad_accumulator))
            embedding_grad_accumulator = embedding_grad + embedding_grad_accumulator

            # reset gradients, just to make sure
            for p_ in curr_layer_perturbation:
                p_.grad.data.zero_()
            curr_embedding_perturbation.grad.data.zero_()

            # remove hidden states from the graph
            new_unpert_layers = []
            for p_ in unpert_layers:
                new_unpert_layers.append(p_.detach())
            unpert_layers = new_unpert_layers
            unpert_embedding = unpert_embedding.detach()

        # apply the accumulated perturbations to the past
        layer_grad_accumulator = [self.to_var(torch.from_numpy(p_), requires_grad=True)
            for p_ in layer_grad_accumulator]
        pert_layers = list(map(add, unpert_layers[:-1], layer_grad_accumulator))
        pert_layers.append(unpert_layers[-1])

        embedding_grad_accumulator = self.to_var(torch.from_numpy(embedding_grad_accumulator), requires_grad=True)
        pert_embedding = unpert_embedding + embedding_grad_accumulator

        return pert_layers, pert_embedding, new_accumulated_hidden, layer_grad_norms, embedding_grad_norm, loss_per_iter

    @torch.enable_grad()
    def perturb_past(
            self,
            # discriminator,
            input_shape,
            token_type_ids,
            position_ids,
            attention_mask,
            task_idx,
            mask_qkv,
            curr_ids,
            next_pos,
            stepsize,
            prev_embedding=None,
            prev_encoded_layers=None,
            new_embedding=None,
            new_encoded_layers=None,
            unpert_logits=None,
            accumulated_hidden=None,
            layer_grad_norms=None,
            embedding_grad_norm=None,
            mask_ids=None,
            sos_ids=None
    ):
        """
            prev_embedding and prev_encoded_layers are obtained from a prev forward pass (FP).    
                prev_embedding: tensor:d_batch * seq_len * d_hidden
                prev_encoded_layers: list of tensors:d_batch * seq_len * d_hidden

            new_embedding and new_encoded_layers are obtained from the current forward pass (FP). 
        """
        def build_past(prev_embedding, prev_encoded_layers, new_embedding, new_encoded_layers):
            """
                Build past for perturbation.

                In the first perturbation, new_* cover {context, [MASK]}, and prev_* are None.
                Therefore, we use new_*[:-1] as past.

                In the later perturbation, prev_* are provided, and we use them as past.

            """
            assert not self.pos_shift
            past = {}
            
            if prev_embedding is None:  # is the first, new_embedding is context + MASK; perturb the context
                past['embedding'] = new_embedding[:, :-1, :]
                past['layers'] = [x[:, :-1, :] for x in new_encoded_layers]
            else:
                past['embedding'] = prev_embedding
                past['layers'] = prev_encoded_layers
                
            return past

        def build_unpert_past(prev_embedding, prev_encoded_layers, new_embedding, new_encoded_layers):
            """
                Build unpert_past for future tokens.

                In the first generation, new_* cover {context, [MASK]}, and prev_* are None.
                We drop the last [MASK], and use new_*[:-1] as unperturb_past.
                
                In the later perturbation, prev_* are provided, and we concat prev_* and new_*[:-1].
            """
            assert not self.pos_shift
            unperturb_past = {}
            
            if prev_embedding is None:  # is the first, new_embedding is context + MASK; perturb the context
                unperturb_past['embedding'] = new_embedding[:, :-1, :]
                unperturb_past['layers'] = [x[:, :-1, :] for x in new_encoded_layers]
            else:
                unperturb_past['embedding'] = torch.cat((prev_embedding, new_embedding[:, :-1, :]), dim=1)
                unperturb_past['layers'] = [
                    torch.cat((prev_encoded_layers[layer_idx], new_encoded_layers[layer_idx][:, :-1, :]), dim=1)
                    for layer_idx in range(len(prev_encoded_layers))]
            
            return unperturb_past

        past = build_past(prev_embedding, prev_encoded_layers, 
            new_embedding, new_encoded_layers)
        unperturb_past = build_unpert_past(prev_embedding, prev_encoded_layers, 
            new_embedding, new_encoded_layers)
        
        grad_accumulator = {}
        grad_accumulator['layers'] = [
            (np.zeros(p.shape).astype("float32"))
            for p in past['layers'][:-1]
        ]
        grad_accumulator['embedding'] = np.zeros(past['embedding'].shape).astype("float32")

        if accumulated_hidden is None:
            accumulated_hidden = 0
        # accumulated_hidden = torch.sum(past['layers'][-1], dim=1)  # sum of the current history
        # print(f'accumulated_hidden: {accumulated_hidden}')

        if self.decay:
            decay_mask = torch.arange(0., 1.0 + SMALL_CONST, 
                1.0 / (self.window_length))[1:]
        else:
            decay_mask = 1.0

        # Generate a mask is gradient perturbated is based on a past window
        # curr_length = grad_accumulator['embedding'].shape[-2]  
        # current history length (context + generation so far)
        curr_length = past['embedding'].shape[-2]
        if self.verbosity_level >= REGULAR:
            print(f'curr_length: {curr_length}')

        if curr_length > self.window_length and self.window_length > 0:
            # FIXME the order of concatenation? should it be zeros first, and then ones?
            d_batch, _, d_hidden = past['embedding'].size()
            ones_key_val_shape = (d_batch, self.window_length, d_hidden)
            zeros_key_val_shape = (d_batch, curr_length - self.window_length, d_hidden)
            
            ones_mask = torch.ones(ones_key_val_shape)
            # ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
            ones_mask = decay_mask * ones_mask.permute(0, 2, 1)
            ones_mask = ones_mask.permute(0, 2, 1)

            window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(self.device)
        else:
            # window_mask = torch.ones_like(past[0]).to(device)
            window_mask = torch.ones_like(past['embedding']).to(self.device)

        # accumulate perturbations for num_iterations
        loss_per_iter = []
        new_accumulated_hidden = None
        for i in range(self.num_iterations):
            if self.verbosity_level >= VERBOSE:
                print(f'\tPerturb Iter {next_pos}.{i + 1}')
            
            curr_layer_perturbation = [
                self.to_var(torch.from_numpy(p_), requires_grad=True)
                for p_ in grad_accumulator['layers']]
            curr_embedding_perturbation = self.to_var(
                torch.from_numpy(grad_accumulator['embedding']), 
                requires_grad=True)

            # Compute hidden using perturbed past
            perturbed_layers = list(map(add, past['layers'][:-1], curr_layer_perturbation))
            perturbed_layers.append(past['layers'][-1])
            pertubed_embedding = past['embedding'] + curr_embedding_perturbation

            step_base_params = {
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'task_idx': task_idx,
                'mask_qkv': mask_qkv,
                'next_pos': next_pos,
                'mask_ids': mask_ids,
                'sos_ids': sos_ids,
            }
            logits, new_embedding, new_encoded_layers = self.step_for_current_perturb(**step_base_params,
                input_shape=input_shape,
                perturbed_embedding=pertubed_embedding, 
                perturbed_layers=perturbed_layers,
                curr_ids=curr_ids
            )

            # hidden = new_encoded_layers[-1] # last hidden layer, for only the current input
            # use only the last token for accumulation
            # TODO: implement another option: use the first token, instead of using [MASK]
            hidden = new_encoded_layers[-1][:, -1, :]  # last hidden state of the last hidden layer
            # print(f'hidden: {hidden.size()}')
            # new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()
            new_accumulated_hidden = accumulated_hidden + hidden.detach()
            
            probs = F.softmax(logits, dim=-1)

            loss = 0.0
            # loss_list = []

            if self.horizon_length == 1:  # obtain future hideen states
                # as input, use the last expected embedding (intead of actual embedding of a token)
                wte = self.bert.embeddings.word_embeddings  # n_vocab * n_hidden
                inputs_embeds = torch.matmul(probs, wte.weight.data)

                next_embedding, next_layers = self.step_for_future_hidden(**step_base_params,
                    input_shape=input_shape,
                    input_embeds=inputs_embeds,
                    unperturb_past=unperturb_past
                )
                # TODO: implement another option is to use the first token, instead of using [MASK]
                next_hidden = next_layers[-1][:, -1, :]  # last hidden state of the last hidden layer
                # print(f'next_hidden: {next_hidden.size()}')
                # new_accumulated_hidden = new_accumulated_hidden + torch.sum(next_hidden, dim=1)
                new_accumulated_hidden = new_accumulated_hidden + next_hidden
            
            elif self.horizon_length > 1:
                raise ValueError('Cannot set horizon_length over 1 since it has not been implemented.')

            # 1 is the perturbation for the present, horizon_length is for the future
            cand_rep = new_accumulated_hidden / (curr_length + 1 + self.horizon_length)
            discrim_loss, group_score, instc_score = self.discriminator(cand_rep)
            if self.verbosity_level >= VERY_VERBOSE:
                print(f'group_score: {group_score[0]}')
                print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            # loss_list.append(discrim_loss)

            kl_loss = 0.0
            if self.kl_scale > 0.0:
                # unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
                unpert_probs = F.softmax(unpert_logits, dim=-1)
                # if self.fp16:
                #     unpert_correction = SMALL_CONST * (unpert_probs <= SMALL_CONST).half().to(self.device).detach()
                #     correction = SMALL_CONST * (probs <= SMALL_CONST).half().to(self.device).detach()
                # else:
                #     unpert_probs = SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(self.device).detach()
                #     correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(self.device).detach()

                # unpert_probs = (unpert_probs + unpert_correction.detach())
                # corrected_probs = probs + correction.detach()
                # print(f'corrected_probs: {corrected_probs}')
                
                # print(f'probs: {probs}')
                # print(f'unpert_probs: {unpert_probs}')
                # div = corrected_probs * (corrected_probs / unpert_probs).log()
                # TODO double check KL loss; it was minus sometimes
                kl_loss_layer = torch.nn.KLDivLoss(reduction='sum')
                # div = kl_loss_layer(probs, unpert_probs)
                div = kl_loss_layer(logits, unpert_probs)
                kl_loss = self.kl_scale * div

                if self.verbosity_level >= VERY_VERBOSE:
                    print(f'kl_loss: {kl_loss.data.cpu().numpy()}')
                loss += kl_loss

            loss_per_iter.append(loss.data.cpu().numpy())
            if self.verbosity_level >= VERBOSE:
                print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

            if self.verbosity_level >= DEBUG:
                discrim_loss.retain_grad()
                group_score.retain_grad()
                cand_rep.retain_grad()
                new_accumulated_hidden.retain_grad()
                # print(f'curr_layer_perturbation 0: {curr_layer_perturbation[0].requires_grad}, {curr_layer_perturbation[0]}')
                # print(f'cand_rep: {cand_rep.requires_grad}, {cand_rep}')
                # print(f'group_score: {group_score.requires_grad}, {group_score}')
                # print(f'discrim_loss: {discrim_loss.requires_grad}, {discrim_loss}')
                # print(f'loss: {loss.requires_grad}, {loss}')

            # compute gradients
            # loss.backward(retain_graph=True)
            loss.backward()

            if self.verbosity_level >= DEBUG:
                print(f'Grad of discrim_loss: {discrim_loss.grad}')
                print(f'Grad of group_score: {group_score.grad}')
                print(f'Grad of cand_rep: {cand_rep.grad}')
                print(f'Grad of new_accumulated_hidden: {new_accumulated_hidden.grad}')
                for index, p_ in enumerate(curr_layer_perturbation):
                    print(f'Grad of curr_layer_perturbation[{index}]: {p_.grad}')

            # calculate gradient norms
            if layer_grad_norms is not None and embedding_grad_norm is not None:
                layer_grad_norms = [
                    torch.max(layer_grad_norms[index], torch.norm(p_.grad * window_mask))
                    for index, p_ in enumerate(curr_layer_perturbation)
                ]
                embedding_grad_norm = torch.max(embedding_grad_norm, torch.norm(curr_embedding_perturbation.grad * window_mask))
            else:
                layer_grad_norms = [
                    (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                    for index, p_ in enumerate(curr_layer_perturbation)
                ]
                embedding_grad_norm = torch.norm(curr_embedding_perturbation.grad * window_mask) + SMALL_CONST

            def finalize_grad(var, norm):
                return -stepsize * (var.grad * window_mask / norm ** self.gamma).data.cpu().numpy()

            layer_grad = [
                finalize_grad(p_, norm=layer_grad_norms[index])
                for index, p_ in enumerate(curr_layer_perturbation)
            ]
            embedding_grad = finalize_grad(curr_embedding_perturbation, norm=embedding_grad_norm)

            # accumulate gradient
            grad_accumulator['layers'] = list(map(add, layer_grad, grad_accumulator['layers']))
            grad_accumulator['embedding'] = embedding_grad + grad_accumulator['embedding']

            # reset gradients, just to make sure
            for p_ in curr_layer_perturbation:
                p_.grad.data.zero_()
            curr_embedding_perturbation.grad.data.zero_()

            # remove hidden states from the graph
            new_past_layers = []
            for p_ in past['layers']:
                new_past_layers.append(p_.detach())
            past['layers'] = new_past_layers
            past['embedding'] = past['embedding'].detach()

        # apply the accumulated perturbations to the past
        pert_past= {}

        grad_accumulator['layers'] = [self.to_var(torch.from_numpy(p_), requires_grad=True)
            for p_ in grad_accumulator['layers']]
        pert_past['layers'] = list(map(add, past['layers'][:-1], grad_accumulator['layers']))
        pert_past['layers'].append(past['layers'][-1])  # the last layer has no gradient

        grad_accumulator['embedding'] = self.to_var(
            torch.from_numpy(grad_accumulator['embedding']), 
            requires_grad=True)
        pert_past['embedding'] = past['embedding'] + grad_accumulator['embedding']

        return pert_past, new_accumulated_hidden, layer_grad_norms, embedding_grad_norm, loss_per_iter

    @torch.enable_grad()
    def step_for_current_perturb(self, 
            input_shape, token_type_ids, position_ids, attention_mask, 
            task_idx=None, mask_qkv=None,
            perturbed_embedding=None, 
            perturbed_layers=None, 
            curr_ids=None, 
            next_pos=None,
            mask_ids=None, 
            sos_ids=None):
        """
            curr_ids: 
                At the first generation step, curr_ids is the conditional context.
                At the later generation steps, curr_ids are the output from the last step.

            We encode x_input_ids.
                If the current step is the first step, x_input_ids is only a [MASK], i.e., input_len=1
                For later steps, we need encode the concatenation of (curr_ids, [MASK]), i.e., input_len=2
        """
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        assert next_pos < output_length

        # loop starts
        # curr_length = list(curr_ids.size())[1]
        def _get_x_input_ids_and_start_pos():
            """
                start_pos: the start pos for the current input
            """
            assert not self.pos_shift, 'Input has not been implemented when pos_shift is True'

            if next_pos == input_length:
                x_input_ids = mask_ids
                start_pos = next_pos 
            else:
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
                start_pos = next_pos - 1
            
            return x_input_ids, start_pos
        
        x_input_ids, start_pos = _get_x_input_ids_and_start_pos()
        if self.verbosity_level >= DEBUG:
            print(f'start_pos: {start_pos}, next_pos: {next_pos}')
        
        # prepare input
        curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
        curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
        curr_position_ids = position_ids[:, start_pos:next_pos + 1]

        if self.verbosity_level >= DEBUG:
            print(f'x_input_ids: {x_input_ids.size()}')
            print(f'curr token_type_ids, attention_mask, position_ids: {curr_token_type_ids.size()}')
            if mask_qkv is not None:
                print(f'mask_qkv: {mask_qkv.size()}')
            if perturbed_embedding is not None:
                print(f'perturbed_embedding: {perturbed_embedding.size()}')
            if perturbed_layers:
                print(f'perturbed_layers: {perturbed_layers[0].size()} * {len(perturbed_layers)}')
        
        new_embedding, new_encoded_layers, _ = self.bert(
                input_ids=x_input_ids, token_type_ids=curr_token_type_ids, position_ids=curr_position_ids, 
                attention_mask=curr_attention_mask,
                output_all_encoded_layers=True, 
                prev_embedding=perturbed_embedding, 
                prev_encoded_layers=perturbed_layers, 
                mask_qkv=mask_qkv)

        # make predictions
        last_hidden = new_encoded_layers[-1][:, -1:, :]
        prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx)
        log_scores = F.log_softmax(prediction_scores, dim=-1)
        
        return log_scores, new_embedding, new_encoded_layers
    
    @torch.enable_grad()
    def step_for_future_hidden(self, 
            input_shape, next_pos, 
            input_embeds, token_type_ids, position_ids, attention_mask, 
            unperturb_past,
            task_idx=None, mask_qkv=None,
            mask_ids=None,
            sos_ids=None):
        """
            For future hidden states in plug and play.

        """
        input_length = input_shape[1]

        def _get_x_input_embeds():
            """
                start_pos: the start pos for the current input
            """
            assert not self.pos_shift, 'Input has not been implemented when pos_shift is True'
            mask_embeddings = self.bert.embeddings.word_embeddings(mask_ids)
            if self.verbosity_level >= DEBUG:
                print(f'[Future perturb] input_embeds: {input_embeds.size()}, mask_embeddings: {mask_embeddings.size()}')
            x_input_embeds = torch.cat((input_embeds, mask_embeddings), dim=1)
            
            return x_input_embeds
          
        x_input_embeds = _get_x_input_embeds()
        # curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
        # curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
        # curr_position_ids = position_ids[:, start_pos:next_pos + 1]
        curr_token_type_ids = token_type_ids[:, next_pos:next_pos + 2]
        curr_attention_mask = attention_mask[:, next_pos:next_pos + 2, :next_pos + 2]
        curr_position_ids = position_ids[:, next_pos:next_pos + 2]
        if self.verbosity_level >= DEBUG:
            print(f'[Future perturb] curr_attention_mask: {curr_attention_mask.size()}')
            print(f'[Future perturb] x_input_embeds: {x_input_embeds.size()}')
            print(f'[Future perturb] unperturb_past.embedding: {unperturb_past["embedding"].size()}')
        new_embedding, new_encoded_layers, _ = self.bert(
                input_embeds=x_input_embeds, 
                token_type_ids=curr_token_type_ids, position_ids=curr_position_ids, attention_mask=curr_attention_mask,
                output_all_encoded_layers=True, 
                prev_embedding=unperturb_past['embedding'], 
                prev_encoded_layers=unperturb_past['layers'], 
                mask_qkv=mask_qkv)
        
        return new_embedding, new_encoded_layers

    def step(self, input_shape, token_type_ids, position_ids, attention_mask, 
            task_idx=None, mask_qkv=None,
            prev_embedding=None, 
            prev_encoded_layers=None, 
            curr_ids=None, 
            next_pos=None,
            mask_ids=None, 
            sos_ids=None):
        """
            We run this function twice in each generation step: once before and once after perturbation.

            One special case is at the first step, when next_pos == input_length.
            Before perturbation, prev_embedding and prev_encoded_layers are both None;
            After perturbation, prev_embedding and prev_encoded_layers are provided from perturbed history,
            we only need to encode [MASK] again.

        """
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        assert next_pos < output_length

        curr_length = list(curr_ids.size())[1]
        def _get_x_input_ids_and_start_pos():
            """
                start_pos: the start pos for the current input
            """
            assert not self.pos_shift, 'Input has not been implemented when pos_shift is True'
            if next_pos == input_length and not (prev_embedding is None or prev_encoded_layers is None):
                x_input_ids = mask_ids
                start_pos = next_pos
            else:
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
                start_pos = next_pos - curr_length
            
            return x_input_ids, start_pos
        
        x_input_ids, start_pos = _get_x_input_ids_and_start_pos()
        
        # prepare input
        curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
        curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
        curr_position_ids = position_ids[:, start_pos:next_pos + 1]

        if self.verbosity_level >= DEBUG:
            print(f'x_input_ids: {x_input_ids.size()}')
            print(f'curr_token_type_ids: {curr_token_type_ids.size()}')
            print(f'curr_position_ids: {curr_position_ids.size()}')
            print(f'curr_attention_mask: {curr_attention_mask.size()}')
            if mask_qkv is not None:
                print(f'mask_qkv: {mask_qkv.size()}')
            if prev_embedding is not None:
                print(f'prev_embedding: {prev_embedding.size()}')
            if prev_encoded_layers:
                print(f'prev_encoded_layers: {prev_encoded_layers[0].size()} * {len(prev_encoded_layers)}')
        
        new_embedding, new_encoded_layers, _ = self.bert(
                input_ids=x_input_ids, token_type_ids=curr_token_type_ids, position_ids=curr_position_ids, 
                attention_mask=curr_attention_mask,
                output_all_encoded_layers=True, 
                prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers, 
                mask_qkv=mask_qkv)

        # make predictions
        last_hidden = new_encoded_layers[-1][:, -1:, :]
        prediction_scores, _ = self.cls(last_hidden, None, task_idx=task_idx)
        log_scores = F.log_softmax(prediction_scores, dim=-1)
        # log_scores = nn.LogSoftmax(dim=-1)(prediction_scores)
        
        return log_scores, new_embedding, new_encoded_layers
    
    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None, 
            # discriminator=None,
            summ_id=None, summ_seg_id=None, summ_mask=None, slot_id=None, slot_mask=None):
        assert self.search_beam_size > 1
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        prev_embedding = None  # embeds for existing tokens (including source tokens and generated tokens)
        prev_encoded_layers = None  # hidden states at each layer for existing tokens
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length  # continueous pos after input
        
        if self.pos_shift:
            sos_ids = input_ids.new(batch_size, 1).fill_(self.sos_id)
        else:
            sos_ids = None

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        layer_grad_norms = None
        embedding_grad_norm = None
        
        loss_in_time = []
        
        query_batch = {
            'summ_id': summ_id,
            'summ_seg_id': summ_seg_id,
            'summ_mask': summ_mask,
            'slot_id': slot_id,
            'slot_mask': slot_mask,
        }
        self.discriminator.init_slot_rep(**query_batch)

        while next_pos < output_length:
            is_first = (prev_embedding is None)  # TODO check if moving forward this line matters 
            # first_token = (next_pos == input_length)
            if self.verbosity_level >= DEBUG:
                print(f'POS {next_pos}: generate original token')
            
            step_params = {
                'input_shape': input_shape,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'prev_embedding': prev_embedding,
                'prev_encoded_layers': prev_encoded_layers,
                'task_idx': task_idx,
                'mask_qkv': mask_qkv,
                'curr_ids': curr_ids,
                'next_pos': next_pos,
                'mask_ids': mask_ids,
                'sos_ids': sos_ids,
            }
            # After beam search, new_embedding and new_encoded_layers will be used to 
            # update prev_embedding and prev_encoded_layers
            unpert_logits, new_embedding, new_encoded_layers = self.step(**step_params)
            
            def get_unpert_last_hidden(new_encoded_layers, prev_encoded_layers, is_first):
                """
                    Get unpert_last_hidden: the layer of all hidden states so far.
                """
                # if next_pos == input_length: # the first step
                if is_first:
                    unpert_last_hidden = new_encoded_layers[-1]
                else:
                    assert prev_encoded_layers is not None
                    unpert_last_hidden = torch.cat(
                        (prev_encoded_layers[-1], new_encoded_layers[-1]), dim=1)
                return unpert_last_hidden
            
            def get_accumulated_hidden(unpert_last_hidden, is_first):
                """
                    Get accumulated_hidden: summation of the past hidden states.
                    
                    We extract accumulated_hidden from unpert_last_hidden, 
                    i.e., the layer of all hidden states so far.

                    At the first step, the last representation should be excluded, i.e., for [MASK].
                    
                    In the rest steps, the last two representations should be excluded,
                    i.e., for the last generation token and [MASK].
                """
                # if next_pos == input_length:
                #     curr_len = 1
                # else:
                #     curr_len = 2
                curr_len = 1 if is_first else 2
                accumulated_hidden = unpert_last_hidden[:, :-curr_len, :]  # exclude the current position
                accumulated_hidden = torch.sum(accumulated_hidden, dim=1)  # sum of the current history
                return accumulated_hidden
            
            unpert_last_hidden = get_unpert_last_hidden(new_encoded_layers, 
                prev_encoded_layers, is_first=is_first)
            
            if next_pos < output_length - 1:
                accumulated_hidden = get_accumulated_hidden(unpert_last_hidden, is_first=is_first)

                # check if we are abowe grad max length
                if next_pos >= self.grad_length:
                    current_stepsize = self.stepsize * 0
                else:
                    current_stepsize = self.stepsize

                perturb_params = {
                    # 'discriminator': discriminator,
                    'input_shape': input_shape,
                    'token_type_ids': token_type_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'task_idx': task_idx,
                    'mask_qkv': mask_qkv,
                    'stepsize': current_stepsize,
                    'curr_ids': curr_ids,
                    'next_pos': next_pos,
                    'prev_embedding': prev_embedding,
                    'prev_encoded_layers': prev_encoded_layers,
                    'new_embedding': new_embedding,
                    'new_encoded_layers': new_encoded_layers,
                    'unpert_logits': unpert_logits,
                    'accumulated_hidden': accumulated_hidden,
                    'layer_grad_norms': layer_grad_norms,
                    'embedding_grad_norm': embedding_grad_norm,
                    'mask_ids': mask_ids,
                    'sos_ids': sos_ids,
                }
                if self.verbosity_level >= DEBUG:
                    print(f'POS {next_pos}: perturb model')
                pert_past, _, layer_grad_norms, embedding_grad_norm, loss_this_iter = self.perturb_past(**perturb_params)
                loss_in_time.append(loss_this_iter)

                if self.verbosity_level >= DEBUG:
                    print(f'POS {next_pos}: foward pass with perturbed history')
                step_params['prev_embedding'] = pert_past['embedding']
                step_params['prev_encoded_layers'] = pert_past['layers']
                pert_logits, pert_embedding, pert_layers = self.step(**step_params)
                # log_scores = pert_logits[:, -1, :] / self.temperature  # + SMALL_CONST
                # log_scores = pert_logits / self.temperature
                # pert_probs = F.softmax(pert_logits, dim=-1)  # vocab distribution from modified model

            # for unpert discrim_loss
            # TODO: implement another option: remove the last [MASK]: unpert_last_hidden[:, :-1, :]
            unpert_discrim_loss, _, _ = self.discriminator(torch.mean(unpert_last_hidden, dim=1))
            if self.verbosity_level >= VERY_VERBOSE:
                print(f"unperturbed discrim loss: {unpert_discrim_loss.data.cpu().numpy()}")

            # Fuse the modified model and original model
            # Original way is to fuse the two distributions (after softmax)
            # Here beam search does not need softmax so we do this with logits
            # log_scores = (pert_logits ** self.gm_scale) * (logits ** (1 - self.gm_scale))
            if next_pos < output_length - 1:
                log_scores = self.gm_scale * pert_logits + (1 - self.gm_scale) * unpert_logits
            else:
                log_scores = unpert_logits

            # proc predictions: forbid pre-defined words; forbid EOS when the min_len is not achieved
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos - input_length + 1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)

            # get topK word ids and scores
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if self.verbosity_level >= DEBUG:
                print(f'log_scores: {log_scores.size()}')
                print(f'kk_scores: {kk_scores.size()}')
            
            if len(total_scores) == 0:  # first token
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                if self.verbosity_level >= DEBUG:
                    print(f'last_eos: {last_eos.size()}')
                    print(f'last_seq_scores: {last_seq_scores.size()}')
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).type_as(kk_scores))
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding)
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]
            else:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding[:, :-1, :])
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x[:, :-1, :]) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                           for x in zip(prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)
                if mask_qkv is not None:
                    mask_qkv = first_expand(mask_qkv)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n - 1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not (
                                self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).to(input_ids.device)
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces
