export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm

export MODEL_PATH=$PROJ_PATH/model/unilm_5
export SPLIT=val
export INPUT_JSON=$PROJ_PATH/data/multinews-250-gold_rank/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CKPT=6000,7500,9000,10500,13000,14500
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
  --max_seq_length 768 \
  --max_tgt_length 400 \
  --batch_size 128 \
  --beam_size 5 \
  --length_penalty 0 \
  --forbid_duplicate_ngrams \
  --mode s2s \
  --forbid_ignore_word ".|<t>|</t>" \
  --min_len 250
