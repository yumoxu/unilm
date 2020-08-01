"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle

from transformers import BertTokenizer, RobertaTokenizer
from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
from transformers.tokenization_bert import whitespace_tokenize
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.utils import load_and_cache_examples
# from transformers import \
#     BertTokenizer, RobertaTokenizer
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.tokenization_minilm import MinilmTokenizer

# from s2s_ft.modeling_decoding import BertModel
import transformers
# from pytorch_transformers import (BertConfig, BertModel, BertTokenizer)

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'minilm': MinilmTokenizer,
    'roberta': RobertaTokenizer,
    'unilm': UnilmTokenizer,
}

class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def add_generation_args(parser):
    """
        Add args for UniLM generation.
    
    """
    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--model_ckpt", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")

    # tokenizer_name
    parser.add_argument("--tokenizer_name", default=None, type=str, required=True, 
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")


def add_discriminator_args(parser):
    """
        Add args for discriminator.
    """
    parser.add_argument('--disc_label', type=float, default=1.0,
                        help='The gold label for discriminator to compute loss against')
    parser.add_argument('--disc_loss_idx', type=int, default=-1,
                        help='The loss index for the discrimintor to use. -1 is for group loss and non-negative number is for instance loss.')
    parser.add_argument('--marge_ckpt_dp', type=str, default=None,
                        help='Checkpoint directory for pretrained MaRGE model.')
    
def add_pp_args(parser):
    """
        Add args for Plug-and-Play.
        
    """
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    # parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")


def parse_args(parser):
    add_generation_args(parser)
    add_discriminator_args(parser)
    add_pp_args(parser)
    
    args = parser.parse_args()
    return args


def set_env(args):
    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    
    return device, n_gpu


def get_input(args):
    input_lines = []
    for line in to_pred:
        input_lines.append(tokenizer.convert_ids_to_tokens(line["source_ids"])[:max_src_length])
    if args.subset > 0:
        logger.info("Decoding subset: %d", args.subset)
        input_lines = input_lines[:args.subset]

    input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
    return input_lines


def load_discriminator(args):
    bert_model_name = 'bert-base-uncased'
    # bert_config = BertConfig.from_pretrained(bert_model_name, num_labels=1, finetuning_task='marge')
    bert_config = transformers.BertConfig()
    bert_model = transformers.BertModel.from_pretrained(bert_model_name, from_tf=bool(False), config=bert_config)
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True, do_basic_tokenize=True, additional_special_tokens=['[SLOT]']) 

    model = MargeDiscriminator(bert_model, pool_func='ls', label=args.disc_label, loss_idx=args.disc_loss_idx)
    # ckpt_dp = path_parser.model_save / f'marge_{marge_config.MARGE_CONFIG_ID}' / f'checkpoint-{marge_config.MARGE_CKPT}'
    checkpoint = torch.load(args.marge_ckpt_dp/'pytorch_model.bin')
    model.load_state_dict(checkpoint['model'])

    # model = nn.DataParallel(model, device_ids=device_ids)
    # print('[load_discriminator] Parallel Data to devices: {}'.format(device_ids))
    model.cuda()
    model.eval()
    return model, tokenizer


def set_tokenizer_and_model(args, discriminator):
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case, 
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_type == "roberta":
        vocab = tokenizer.encoder
    else:
        vocab = tokenizer.vocab

    tokenizer.model_max_length = args.max_seq_length
    config_file = args.config_path if args.config_path else os.path.join(args.model_path, "config.json")
    logger.info("Read decoding config from: %s" % config_file)
    config = BertConfig.from_json_file(config_file)

    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
        list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        max_tgt_length=args.max_tgt_length, pos_shift=args.pos_shift,
        source_type_id=config.source_type_id, target_type_id=config.target_type_id, 
        cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token))

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))

    model_path = os.path.join(args.model_path, 'ckpt-{}'.format(args.model_ckpt))
    print(model_path)
    model_recover_path = model_path.strip()
    logger.info("***** Recover model: %s *****", model_recover_path)
    model = BertForQueryFocusedDecoder.from_pretrained(
        model_recover_path, config=config, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
        length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
        forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
        ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
        max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift, 
        discriminator=discriminator,
        device=device,
        verbosity_level=args.verbosity_level, stepsize=args.stepsize, temperature=args.temperature, top_k=args.top_k,
        num_iterations=args.num_iterations, grad_length=args.grad_length, horizon_length=args.horizon_length, 
        window_length=args.window_length, decay=args.decay, gamma=args.gamma, kl_scale=args.kl_scale
    )

    if args.fp16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    torch.cuda.empty_cache()
    model.eval()
    return tokenizer, model, bi_uni_pipeline, model_recover_path


def main():
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    device, n_gpu = set_env(args)
    discriminator, _ = load_discriminator(args)

    # tokenizer, config, bi_uni_pipeline, mask_word_id, eos_word_ids, sos_word_id, forbid_ignore_set = set_tokenizer(args)
    tokenizer, model, bi_uni_pipeline, model_recover_path = set_tokenizer_and_model(args, discriminator=discriminator)

    next_i = 0
    max_src_length = args.max_seq_length - 2 - args.max_tgt_length

    to_pred = load_and_cache_examples(args.input_file, tokenizer, local_rank=-1, cached_features_file=None, shuffle=False)
    input_lines = get_input(args)
    
    output_lines = [""] * len(input_lines)
    score_trace_list = [None] * len(input_lines)
    total_batch = math.ceil(len(input_lines) / args.batch_size)

    with tqdm(total=total_batch) as pbar:  
        batch_count = 0
        first_batch = True
        while next_i < len(input_lines):  # for each batch
            _chunk = input_lines[next_i:next_i + args.batch_size]
            buf_id = [x[0] for x in _chunk]  # line index
            buf = [x[1] for x in _chunk]   # line
            next_i += args.batch_size
            batch_count += 1
            max_a_len = max([len(x) for x in buf])
            instances = []  # processed input lines
            for instance in [(x, max_a_len) for x in buf]:
                for proc in bi_uni_pipeline:   # can ignore this loop; there is only one Preprocess4Seq2seqDecoder in the pipeline
                    instances.append(proc(instance))
            with torch.no_grad():
                batch = seq2seq_loader.batch_list_to_batch_tensors(instances)
                batch = [t.to(device) if t is not None else None for t in batch]
                input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                traces = model(input_ids, token_type_ids, position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces.tolist()
                for i in range(len(buf)):
                    w_ids = output_ids[i]
                    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in (tokenizer.sep_token, tokenizer.pad_token):
                            break
                        output_tokens.append(t)
                    if args.model_type == "roberta":
                        output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                    else:
                        output_sequence = ' '.join(detokenize(output_tokens))
                    if '\n' in output_sequence:
                        output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                    output_lines[buf_id[i]] = output_sequence
                    if first_batch or batch_count % 50 == 0:
                        logger.info("{} = {}".format(buf_id[i], output_sequence))
                    if args.need_score_traces:
                        score_trace_list[buf_id[i]] = {
                            'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
            pbar.update(1)
            first_batch = False
    if args.output_file:
        fn_out = args.output_file
    else:
        fn_out = model_recover_path+'.'+args.split
    with open(fn_out, "w", encoding="utf-8") as fout:
        for l in output_lines:
            fout.write(l)
            fout.write("\n")

    if args.need_score_traces:
        with open(fn_out + ".trace.pickle", "wb") as fout_trace:
            pickle.dump(
                {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
            for x in score_trace_list:
                pickle.dump(x, fout_trace)


if __name__ == "__main__":
    main()
