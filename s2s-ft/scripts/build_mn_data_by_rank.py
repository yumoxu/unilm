import io
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
from tqdm import tqdm
import json
import nltk

SHIFTSUM_ROOT = Path('/home/s1617290/shiftsum/data/multinews')
UNILM_ROOT = Path('/home/s1617290/unilm/data')

FINAL_DATA_DIR_NAME = 'multinews'
TGT_MIN_WORDS = None
if TGT_MIN_WORDS:
    FINAL_DATA_DIR_NAME += f'-{TGT_MIN_WORDS}'

RANK_MODE = 'gold'
METRIC = 'rouge_2_f1'  # rouge_2_recall, rouge_2_f1

FINAL_DATA_DIR_NAME += f'-{RANK_MODE}_rank_{METRIC}'

PREPEND_LEN = True
if PREPEND_LEN:
    FINAL_DATA_DIR_NAME += '_prepend_len'

FINAL_DATA_DIR = UNILM_ROOT / FINAL_DATA_DIR_NAME

DATASET_VAR = 'val' 


if not exists(FINAL_DATA_DIR):
    os.mkdir(FINAL_DATA_DIR)


def get_cid2summary():
    masked_summary_fp = SHIFTSUM_ROOT / 'masked_mn_summary' / f'{DATASET_VAR}-sample-max_reveal_1.0.json'
    cid = 0
    cid2summary = {}
    with open(masked_summary_fp) as masked_summary_f:
        for line in masked_summary_f:
            json_obj = json.loads(line)
            # cid2summary[cid] = {
            #     'masked_seq': json_obj['masked_seq'],
            #     'original_summary': json_obj['original_summary'],
            # }
            cid2summary[cid] = json_obj['original_summary']
            cid += 1
    return cid2summary


def _get_cid(json_obj):
    return int(json_obj['sid'].split('_')[0])


def _rank_sentence_objs(sentence_objs, metric):
    return sorted(sentence_objs, key=lambda so: so[metric], reverse=True)


def get_len_token(tgt_len):
    if tgt_len < 100:
        tgt_len = 85
    elif tgt_len >= 400:
        tgt_len = 400
    else:
        for start in range(100, 400, 15):
            if start <= tgt_len < start+15:
                tgt_len = start
                break
    
    assert (tgt_len-100)%15==0 or tgt_len==85, f'{tgt_len} is not right'
    return f'[unused{tgt_len}]'


def unit_test_get_len_token():
    tgt_lens = [99, 100, 101, 201, 250, 399, 400, 401]
    for tl in tgt_lens:
        token = get_len_token(tl)
        print(f'{tl}\t{token}')


def to_save(tgt_len):
    to_save = True
    if TGT_MIN_WORDS and tgt_len < TGT_MIN_WORDS:
        to_save = False
    
    return to_save


def build():
    rouge_fp = SHIFTSUM_ROOT / 'rouge' / f'{DATASET_VAR}.json'
    cid2summary = get_cid2summary()

    dump_fp = FINAL_DATA_DIR / f'{DATASET_VAR}.json'

    cid = 0
    sentence_objs = []
    with open(dump_fp, 'a') as dump_f:
        with open(rouge_fp) as rouge_f:
            for line in rouge_f:
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)
                if _cid != cid:
                    ranked_sentence_objs = _rank_sentence_objs(sentence_objs, metric=METRIC)

                    if cid % 1000 == 0:
                        print(f'cid: {cid}, #Sentences: {len(sentence_objs)}')
                    
                    tgt = cid2summary[cid]
                    tgt_words = nltk.tokenize.word_tokenize(tgt)
                    tgt_len = len(tgt_words)

                    if to_save(tgt_len):
                        sentences = [so['sentence'].replace('NEWLINE_CHAR', '').strip()
                            for so in ranked_sentence_objs]
                        src = ' '.join(sentences)

                        if PREPEND_LEN:
                            tgt = get_len_token(tgt) + ' ' + tgt
                        
                        dump_obji = {
                            "sentences": ranked_sentence_objs,
                            "src": src,
                            "tgt": tgt,
                        }
                        json_str = json.dumps(dump_obj, ensure_asci=False)
                        dump_f.write(f'{json_str}\n')

                    sentence_objs = []

                so = {
                    'id': json_obj['sid'],
                    'sentence': json_obj['sentence'],
                    'rouge_2_recall': json_obj['rouge_2_recall'],
                    'rouge_2_f1': json_obj['rouge_2_f1'],
                }
                sentence_objs.append(so)
                cid = _cid
    
    print(f'Sucessfully dump {DATASET_VAR} set to: {dump_fp}!')


if __name__ == "__main__":
    build()
    # unit_test_get_len_token()
