export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm

# RR-SUBQ: rr_subq-narr-rr_records-rr-34_config-25000_iter-query-ir-dial-tf-2007-150_qa_topK-prepend_len
# RR: rr-34_config-25000_iter-query-ir-dial-tf-2006-top150-prepend_len
# QA: qa-bert-narr-ir-dial-tf-2007-top90-prepend_len-prepend_masked_q
# QA-tdqfs: qa-bert-narr-ir-dial-tf-tdqfs-top90-prepend_len-prepend_add_left_mask_right_dot_q
# QA-DUC: qa-bert-narr-ir-dial-tf-2006-top90-prepend_len-prepend_masked_q

# RR-with query: 
# rr-34_config-25000_iter-query-ir-dial-tf-2007-top150-prepend_len-prepend_raw_q
# rr-34_config-25000_iter-query-ir-dial-tf-2006-top150-prepend_len-prepend_masked_q
# rr-34_config-25000_iter-query-ir-dial-tf-2007-top150-prepend_len-prepend_masked_q

# rr-39_config-26000_iter-query-ir-dial-tf-2007-top150-prepend_len-prepend_masked_q

# rr-39_config-26000_iter-query-ir-dial-tf-2006-top150-prepend_len

# rr-39_config-26000_iter-add_left_mask_right_dot-ir-dial-tf-tdqfs-top150-prepend_len-prepend_masked_q
# rr-39_config-26000_iter-add_left_mask_right_dot-ir-dial-tf-tdqfs-0.6_cos-nw_250-prepend_len-prepend_masked_q
# rr-39_config-26000_iter-add_left_mask_right_dot-ir-dial-tf-tdqfs-0.6_cos-nw_250-prepend_len-prepend_masked_q
# rr-34_config-25000_iter-pred@grsum-tdqfs-0.6_cos-0_wan-nw_250@masked-ratio-reveal_1.0-ir-dial-tf-tdqfs-top150-prepend_len-prepend_masked_q
# rr-34_config-25000_iter-pred@grsum-tdqfs-0.6_cos-0_wan-nw_250@masked-ratio-reveal_1.0-ir-dial-tf-tdqfs-top150-prepend_len-prepend_add_left_mask_right_dot_q
export INPUT_FILE_NAME=qa-bert-narr-ir-dial-tf-2006-top90-prepend_len-prepend_masked_q
export SPLIT=${INPUT_FILE_NAME}
export MODEL_PATH=$PROJ_PATH/model/unilm_17
export QFS_PROJ_ROOT=/disk/nfs/ostrom/s1617290/shiftsum
export INPUT_JSON=${QFS_PROJ_ROOT}/unilm_in/unilm_in-${INPUT_FILE_NAME}.json

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 500,1000,1500,2000,2500,3000,3500,4000,4500
# 5000,5500,6000,6500,7000,7500,8000,8500,9000, 9500,10000
export CKPT=2500
export python=$PROJ_PATH/bin/python
export python_file=$PROJ_PATH/s2s-ft/decode_seq2seq.py

$python $python_file \
  --fp16 --model_type unilm \
  --tokenizer_name unilm1.2-base-uncased \
  --do_lower_case \
  --prepend_len \
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