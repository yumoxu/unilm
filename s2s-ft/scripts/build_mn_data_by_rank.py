import io
from pathlib import Path
from tqdm import tqdm
import json
import nltk

SHIFTSUM_ROOT = Path('/home/s1617290/shiftsum/data/multinews')
UNILM_ROOT = Path('/home/s1617290/unilm/data')

FINAL_DATA_DIR_NAME = 'multinews'
TGT_MIN_WORDS = 250
if TGT_MIN_WORDS:
    FINAL_DATA_DIR_NAME += f'-{TGT_MIN_WORDS}'

RANK_MODE = 'gold'
FINAL_DATA_DIR_NAME += f'-{RANK_MODE}_rank'
FINAL_DATA_DIR = UNILM_ROOT / FINAL_DATA_DIR_NAME

DATASET_VAR = 'val'


def get_cid2summary():
    masked_summary_fp = SHIFTSUM_ROOT / 'masked_mn_summary' / f'{DATASET_VAR}.json'
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


def build():
    rouge_fp = SHIFTSUM_ROOT / 'multinews_rouge' / f'{DATASET_VAR}.json'
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
                    ranked_sentence_objs = _rank_sentence_objs(sentence_objs, metric=SELECTION_METRIC)

                    if cid % 500 == 0:
                        logger.info(f'cid: {cid}, #Sentences: {len(sentence_objs)}')
                    
                    tgt = cid2summary[cid]

                    to_save = True
                    if TGT_MIN_WORDS and len(nltk.tokenize.word_tokenize(tgt)) >= TGT_MIN_WORDS:
                        to_save = False

                    if to_save:
                        sentences = [so['sentence'] for so in ranked_sentence_objs]
                        src = ' '.join(sentences)
                        json_obj = {
                            "sentences": ranked_sentence_objs,
                            "src": src,
                            "tgt": tgt,
                        }
                        json_str = json.dumps(json_obj, ensure_ascii=False)
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
    
    logger.info(f'Sucessfully dump {DATASET_VAR} set to: {dump_fp}!')


if __name__ == "__main__":
    build()
