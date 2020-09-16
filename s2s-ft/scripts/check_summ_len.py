import io
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize

SHIFTSUM_ROOT = P_ath('/home/s1617290/shiftsum/data/multinews')
UNILM_ROOT = Path('/home/s1617290/unilm')

FINAL_DATA_DIR_NAME = 'multinews'
TGT_MIN_WORDS = None
if TGT_MIN_WORDS:
    FINAL_DATA_DIR_NAME += f'-{TGT_MIN_WORDS}'

RANK_MODE = 'gold'
METRIC = 'rouge_2_recall'  # rouge_2_recall, rouge_2f1

FINAL_DATA_DIR_NAME += f'-{RANK_MODE}_rank_{METRIC}'
FINAL_DATA_DIR = UNILM_ROOT / 'data' / FINAL_DATA_DIR_NAME

DATASET_VAR = 'val' 
# PROJ_ROOT = Path('/disk/nfs/ostrom/s1617290/unilm')
MODEL_DIR = UNILM_ROOT / 'model' / 'unilm1.2-base-uncased'
CACHE_DIR = UNILM_ROOT / 'model' / 'cache_unilm'
STATS_DIR - UNILM_ROOT / 'stats'


if not exists(FINAL_DATA_DIR):
    os.mkdir(FINAL_DATA_DIR)


def get_cid2summary():
    masked_summary_fp = SHIFTSUM_ROOT / 'masked_mn_summary' / f'{DATASET_VAR}-sample-max_reveal_1.0.json'
    cid = 0
    cid2summary = {}
    with open(masked_summary_fp) as masked_summary_f:
        for line in masked_summary_f:
            json_obj = json.loads(line)
            cid2summary[cid] = json_obj['original_summary']
            cid += 1
    return cid2summary


def get_tokenizer():
    tokenizer_name = 'unilm1.2-base-uncased'
    model_name_or_path = MODEL_DIR
    do_lower_case = True
    cache_dir = CACHE_DIR
    from s2s_ft.tokenization_unilm import UnilmTokenizer
    tokenizer = UnilmTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case, cache_dir=cache_dir)

    
def _draw(n_token, range, n_bins, xlabel, title, color='darkblue'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_token = [int(nt) for nt in n_token]
    counts, bin_edges = np.histogram(n_token, bins=n_bins, range=range, density=False)
    dist = [c/float(sum(counts)) for c in counts]

    logger.info(f'total counts: {sum(counts)}')
    logger.info(f'distribution: {dist}')
    logger.info(f'bin_edges: {bin_edges}')

    fig = plt.figure(figsize=(5, 4))
    sns.distplot(n_token, hist=True, kde=True, 
        bins=n_bins, 
        color=color, 
        hist_kws={'edgecolor':'black', 'range': range}, 
        kde_kws={'linewidth': 2})
    # plt.hist(np.array(n_words), bins=7, range=(10, 80), density=True, stacked=True)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.tight_layout()
    
    fig.savefig(DP_PROJ/'stats'/ title, bbox_inches='tight')
    plt.show()


def summary_stats():
    cid2summary = get_cid2summary()
    tokenizer = get_tokenizer()

    dump_fp = FINAL_DATA_DIR / f'{DATASET_VAR}.json'
    stat_fp = STATS_DIR / f'stats_{DATASET_VAR}_{FINAL_DATA_DIR_NAME}.txt'

    if not exists(stat_fp):
        logger.info('Build stat file')
        with io.open(stat_fp, mode='a') as stat_f:
            stat_f.write('n_token\tn_words\n')

            for cid, summary in cid2summary.items():
                n_token = len(tokenizer(summary))
                n_word =len(word_tokenize(summary))
                stat_f.write(f'{n_token}\t{n_word}\n')

    logger.info(f'Read from stat fille: {stat_fp}')
    with io.open(stat_fp) as stat_f:
        lines = stat_f.readlines()[1:]
        items = [line.strip('\n').split('\t') for line in lines]
        n_token, n_word = zip(*items)
        _draw(n_token=n_token, range=[100, 500], n_bins=400, coor='darkblue',
            xlabel='Number of tokens', title=f'token_dist_{DATASET_VAR}_{FINAL_DATA_DIR_NAME}.pdf')


if __name__ == "__main__":
    summary_stats()
