export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm

# marge-13_config-37500_iter-narr-ir-dial-tf-2005-top90-global_pos
# rr-34_config-25000_iter-query-ir-dial-tf-2007-top150-local_pos
# rr-34_config-25000_iter-pred@grsum-tdqfs-0.6_cos-0_wan-nw_250@masked-ratio-reveal_1.0-ir-dial-tf-tdqfs-top150-local_pos
export INPUT_FILE_NAME=rr-34_config-25000_iter-query-ir-dial-tf-2006-top150-local_pos
export SPLIT=${INPUT_FILE_NAME}
export MODEL_PATH=$PROJ_PATH/model/unilm_20
export QFS_PROJ_ROOT=/disk/nfs/ostrom/s1617290/shiftsum
export INPUT_JSON=${QFS_PROJ_ROOT}/unilm_in/unilm_in-${INPUT_FILE_NAME}.json

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export CKPT=1500,3000,4500,6000,7500,9000,10500,12000,13500,15000,16500,18000,19500,21000,22500,24000,25500,27000,28500,30000,31500,33000,34500,36000,37500,39000
export python=$PROJ_PATH/bin/python
export python_file=$PROJ_PATH/s2s-ft/decode_seq2seq.py

$python $python_file \
  --fp16 --model_type unilm \
  --tokenizer_name unilm1.2-base-uncased \
  --do_lower_case \
  --input_file ${INPUT_JSON} \
  --split ${SPLIT} \
  --model_path ${MODEL_PATH} \
  --model_ckpt ${CKPT} \
  --max_seq_length 1168 \
  --max_tgt_length 400 \
  --batch_size 32 \
  --beam_size 5 \
  --length_penalty 0.9 \
  --forbid_duplicate_ngrams \
  --mode s2s \
  --forbid_ignore_word "." \
  --min_len 300