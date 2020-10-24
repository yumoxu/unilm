import io
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
from tqdm import tqdm
import json
import nltk

"""
    This file buids training/dev data from CNN/DM clusters for UniLM.

    Required:
        - Cluster json containing cluster info, e.g., sample ids
        - ROUGE json containing sentence info, based on which we produce sentence rank for each cluster

    Pipeline
        - Build docs:
            Output: train-doc.json, which contains ranked sentence objects for each doc
        - Build clusters:
            Output: train-cluster.json, which containis merged ranked lists for docs
    
"""
SHIFTSUM_ROOT = Path('/home/s1617290/shiftsum/data/cnndm')
UNILM_ROOT = Path('/home/s1617290/unilm/data')

FINAL_DATA_DIR_NAME = 'cnndm'
TGT_MIN_WORDS = None
if TGT_MIN_WORDS:
    FINAL_DATA_DIR_NAME += f'-{TGT_MIN_WORDS}'

RANK_MODE = 'gold'
METRIC = 'rouge_2_f1'  # rouge_2_recall, rouge_2_f1

SHORT_METRIC = METRIC.split('_')[-1]
FINAL_DATA_DIR_NAME += f'-{RANK_MODE}_rank_{SHORT_METRIC}'

ROUGE_C = 0.15
SMOOTH_METRIC = f'rouge_1_{SHORT_METRIC}'
if ROUGE_C > 0:
    FINAL_DATA_DIR_NAME += f'_{ROUGE_C}'

PREPEND_LEN = False
if PREPEND_LEN:
    FINAL_DATA_DIR_NAME += '_prepend_len'

SWAP_PROB = 0.0
if SWAP_PROB > 0.0:
    FINAL_DATA_DIR_NAME += '_{SWAP_PROB}_swap'

FINAL_DATA_DIR = UNILM_ROOT / FINAL_DATA_DIR_NAME

DATASET_VAR = 'val' 
CLUSTER_FN = f'cluster-{DATASET_VAR}-cos_0.6.json'

if not exists(FINAL_DATA_DIR):
    os.mkdir(FINAL_DATA_DIR)

DOC_DUMP_FP = FINAL_DATA_DIR / f'{DATASET_VAR}-doc.json'
CLUSTER_DUMP_FP = FINAL_DATA_DIR / f'{DATASET_VAR}-cluster.json'


def load_cluster():
    cluster_fp = SHIFTSUM_ROOT / 'clusters' / CLUSTER_FN
    cid2info = {}
    cids = []
    with open(masked_summary_fp) as cluster_f:
        for line in cluster_f:
            json_obj = json.loads(line)
            cid = json_obj['cluster_id']
            summary = proc_summary(json_obj['cluster_summaries'])
            
            cid2info[cid] = {
                'doc_ids': json_obj['cluster_sample_ids'],
                'summary': summary,
            }
            cids.append(cid)

    return cid2info, cids


def proc_summary(summaries):
    return ' '.join(summaries)


class Doc:
    def __init__(self, sample_id, ranked_sentence_objs):
        self.sample_id = sample_id
        self.ranked_sentence_objs = ranked_sentence_objs


def _get_did(json_obj):
    return int(json_obj['sid'].split('_')[0])


def _rank_sentence_objs(sentence_objs, metric, rouge_c, smooth_metric):
    if rouge_c > 0.0:
        for so in sentence_objs:
            smoothed_score = (1 - rouge_c) * float(so[metric]) + rouge_c * float(so[smooth_metric])
            so['smoothed_score'] = smoothed_score
        ranked_objs = sorted(sentence_objs, key=lambda so: so['smoothed_score'], reverse=True)
    else:
        ranked_objs = sorted(sentence_objs, key=lambda so: so[metric], reverse=True)

    return ranked_objs


def _swap_sentence_objs(sentence_objs, metric, swap_prob):
    def _swap(i, j):
        temp = sentence_objs[j]
        sentence_objs[j] = sentence_objs[i]
        sentence_objs[i] = temp
        
    status = [0] * len(sentence_objs)

    for ii, so in enumerate(sentence_objs[:-1]):
        if status[ii] == 1:  # has bee swapped
            continue
            
        do_swap = np.random.choice([0, 1], p=np.array([1.0-swap_prob, swap_prob]))
        if not do_swap:
            continue
        
        candidates = sentence_objs[ii+1:]
        for relative_pos, _so in enumerate(candidates):
            score = 1.0 / (math.abs(so[metric]-_so[metric]) + 1e-7)
            if status[jj] == 1:
                score = 0.0
            scores.append(score)
        nom = sum(scores)

        prob_dist = np.array([sc/nom for sc in scores])
        indices = ii + 1 + np.arange(len(prob_dist))
        jj = np.random.choice(indices, p=prob_dist)  # abs_pos
        _swap(ii, jj)


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


def to_save(tgt_len):
    to_save = True
    if TGT_MIN_WORDS and tgt_len < TGT_MIN_WORDS:
        to_save = False
    
    return to_save


def sentence_objs2records(sentence_objs, doc_id):
    ranked_sentence_objs = _rank_sentence_objs(sentence_objs, 
        metric=METRIC, rouge_c=ROUGE_C, smooth_metric=SMOOTH_METRIC)
    if SWAP_PROB > 0.0:
        _swap_sentence_objs(sentence_objs, metric=METRIC, swap_prob=SWAP_PROB)

    if doc_id % 1000 == 0:
        print(f'doc id: {doc_id}, #Sentences: {len(sentence_objs)}')
    
    dump_obj = {
        "doc_id": doc_id,
        "sentences": ranked_sentence_objs,
    }
    json_str = json.dumps(dump_obj)
    return json_str

    
def build_docs():
    """
        Rank sentences per sample, and dump them.

    """
    rouge_fp = SHIFTSUM_ROOT / 'rouge' / f'{DATASET_VAR}.json'

    doc_id = 0
    sentence_objs = []
    with open(DOC_DUMP_FP, 'a') as dump_f:
        with open(rouge_fp) as rouge_f:
            for line in rouge_f:
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _doc_id =  _get_did(json_obj)
                if _doc_id != doc_id:
                    if doc_id % 1000 == 0:
                        print(f'doc id: {doc_id}, #Sentences: {len(sentence_objs)}')

                    json_str = sentence_objs2records(sentence_objs, doc_id)
                    dump_f.write(f'{json_str}\n')
                    sentence_objs = []

                so = {
                    'id': json_obj['sid'],
                    'sentence': json_obj['sentence'],
                    'rouge_1_recall': json_obj['rouge_1_recall'],
                    'rouge_1_f1': json_obj['rouge_1_f1'],
                    'rouge_2_recall': json_obj['rouge_2_recall'],
                    'rouge_2_f1': json_obj['rouge_2_f1'],
                }
                sentence_objs.append(so)
                doc_id = _doc_id
            
            if sentence_objs:  # finish the last
                json_str = sentence_objs2records(sentence_objs, doc_id)
                dump_f.write(f'{json_str}\n')
    
    print(f'Sucessfully dump {DATASET_VAR} set to: {DOC_DUMP_FP}')


def load_docs():
    doc_id2rank = {}
    with open(DOC_DUMP_FP) as doc_f:
        for line in tqdm(doc_f):
            json_obj = json.loads(line)
            doc_id2rank[json_obj['doc_id']] = json_obj['sentences']

    return doc_id2rank


def merge(cluster_sentences):
    """
        # TODO merge multiple sorted lists
    """
    pass


def build_clusters():
    """
        Rank sentences per clusters, and dump them.
        
    """
    cid2info, cids = load_cluster()
    doc_id2rank = load_docs()

    with open(CLUSTER_DUMP_FP, 'a') as dump_f:
        for cid in cids:
            if cid % 1000 == 0:
                print(f'cid: {cid}, #Sentences: {len(sentence_objs)}')
            
            # TODO get sentence_objs from doc_id2rank
            cluster_info = cid2info[cid]
            doc_ids = cluster_info['doc_ids']
            cluster_sentences = [doc_id2rank[doc_id] 
                for doc_id in doc_ids]
            
            ranked_sentence_objs = merge(cluster_sentences)

            tgt = cluster_info['summary']
            tgt_words = nltk.tokenize.word_tokenize(tgt)
            tgt_len = len(tgt_words)
            if to_save(tgt_len):
                sentences = [so['sentence'].replace('NEWLINE_CHAR', '').strip()
                    for so in ranked_sentence_objs]
                src = ' '.join(sentences)
                
                if PREPEND_LEN:
                    src = get_len_token(tgt_len) + ' ' + src
                
                dump_obj = {
                    "sentences": ranked_sentence_objs,
                    "src": src,
                    "tgt": tgt,
                }
                # json_str = json.dumps(dump_obj, ensure_asci=False)
                json_str = json.dumps(dump_obj)
                dump_f.write(f'{json_str}\n')
    
    print(f'Sucessfully dump {DATASET_VAR} set to: {CLUSTER_DUMP_FP}')


if __name__ == "__main__":
    # unit_test_get_len_token()
    # unit_test_swap_sentence_objs()
    # build_docs()
    build_clusters()
