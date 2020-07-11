import io
from pathlib import Path
from tqdm import tqdm
import json

DATA_ROOT = Path('/Users/KevinXU/Programming/git_yumoxu/unilm/data')
TRUNCATED_DATA_ROOT = DATA_ROOT / 'multinews_truncated'
FINAL_DATA_DIR = DATA_ROOT / 'multinews'
DATASET_VAR = 'test'


def build():
    src_fp = TRUNCATED_DATA_ROOT / f'{DATASET_VAR}.txt.src.tokenized.fixed.cleaned.final.truncated'
    tgt_fp = TRUNCATED_DATA_ROOT / f'{DATASET_VAR}.txt.tgt.tokenized.fixed.cleaned.final.truncated'
    dump_fp = FINAL_DATA_DIR / f'{DATASET_VAR}.json'

    src_text = io.open(src_fp).readlines()
    tgt_text = io.open(tgt_fp).readlines()
    with io.open(dump_fp, mode='a') as dump_f:
        for i in tqdm(range(len(src_text))):
            json_obj = {
                "src": src_text[i].strip('\n'),
                "tgt": tgt_text[i].strip('\n')[2:]
            }
            json_str = json.dumps(json_obj, ensure_ascii=False)
            dump_f.write(f'{json_str}\n')

if __name__ == "__main__":
    build()
