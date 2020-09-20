export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm

export MODEL_PATH=$PROJ_PATH/model/unilm_11
export SPLIT=val
export INPUT_JSON=$PROJ_PATH/data/multinews-gold_rank_f1/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 1500,3000,4500,6000,7500,9000,10500,12000,13500,15000,16500,18000,19500
# 1500,3000,4500,6000,7500,9000,10500
# 12000,13500,15000,16500,18000,19500
export CKPT=1500,3000,4500,6000,7500,9000,10500
export python=$PROJ_PATH/bin/python
export python_file=$PROJ_PATH/s2s-ft/decode_seq2seq.py

$python $python_file \
  --fp16 --model_type unilm \
  --tokenizer_name unilm1.2-base-uncased \
  --do_lower_case \
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
