export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm

export MODEL_PATH=$PROJ_PATH/model/unilm_1
export SPLIT=cnndm.dev.uncased_tokenized
export INPUT_JSON=$PROJ_PATH/data/cnndm/${SPLIT}.json

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CKPT=33000,31500,30000,28500,27000,25500,24000,22500,21000,19500
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
  --max_tgt_length 160 \
  --batch_size 128 \
  --beam_size 5 \
  --length_penalty 0 \
  --forbid_duplicate_ngrams \
  --mode s2s \
  --forbid_ignore_word "." \
  --min_len 48
