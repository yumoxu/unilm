export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm

export MODEL_PATH=$PROJ_PATH/model/unilm_12
export SPLIT=val
export INPUT_JSON=$PROJ_PATH/data/rouge_estimated-rr-34_config-25000_iter-masked_summ-mn-marge_cluster-ratio-reveal_0.0_prepend_len/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# snippy: 1500,3000,4500,6000
# quarry: 7500,9000,10500,12000
# ostrom: 13500,15000,16500,18000
# nuess: 19500, 21000, 22500, 24000
# duflo: 25500, 27000, 28500, 30000, 31500, 33000, 34500, 36000, 37500, 39000
export CKPT=1500,3000,4500,6000
export python=$PROJ_PATH/bin/python
export python_file=$PROJ_PATH/s2s-ft/decode_seq2seq.py

$python $python_file \
  --fp16 --model_type unilm \
  --tokenizer_name unilm1.2-base-uncased \
  --do_lower_case \
  --prepend_len \
  --input_file ${INPUT_JSON} \
  --split $SPLIT \
  --model_path ${MODEL_PATH} \
  --model_ckpt ${CKPT} \
  --max_seq_length 1068 \
  --max_tgt_length 300 \
  --batch_size 96 \
  --beam_size 5 \
  --length_penalty 0 \
  --forbid_duplicate_ngrams \
  --mode s2s \
  --forbid_ignore_word ".|<t>|</t>" \
  --min_len 200
