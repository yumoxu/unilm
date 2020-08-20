export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm
export SPLIT=qfs-step_1.0
export UNILM_MODEL_PATH=$PROJ_PATH/model/unilm_4
export UNILM_CKPT=4500

export QFS_PROJ_ROOT=/disk/nfs/ostrom/s1617290/shiftsum
export MARGE_MODEL_NAME=marge_13
export MARGE_CKPT=37500
export MARGE_CKPT_PATH=${QFS_PROJ_ROOT}/model/${MARGE_MODEL_NAME}/checkpoint-${MARGE_CKPT}
export INPUT_JSON=${QFS_PROJ_ROOT}/unilm_in/unilm_in-marge-13_config-37500_iter-narr-ir-dial-tf-2005-top50-global_pos.json

export python=$PROJ_PATH/bin/python
export python_file=$PROJ_PATH/s2s-ft/qfs_decode.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export OMP_NUM_THREADS=2
# export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

$python $python_file \
  --fp16 --model_type unilm \
  --tokenizer_name unilm1.2-base-uncased \
  --do_lower_case \
  --input_file ${INPUT_JSON} \
  --split ${SPLIT} \
  --model_path ${UNILM_MODEL_PATH} \
  --model_ckpt ${UNILM_CKPT} \
  --max_seq_length 1168 \
  --max_tgt_length 400 \
  --batch_size 16 \
  --beam_size 5 \
  --length_penalty 0.9 \
  --forbid_duplicate_ngrams \
  --mode s2s \
  --forbid_ignore_word "." \
  --min_len 300 \
  --marge_ckpt_dp ${MARGE_CKPT_PATH}\
  --year='2005' \
  --max_summ_seq_len=96 \
  --max_num_slot=32 \
  --add_cls_at_begin \
  --disc_label 1.0 \
  --disc_loss_idx -1 \
  --gamma 1.0 \
  --num_iterations 5 \
  --horizon_length 1 \
  --stepsize 1.0 \
  --kl_scale 0.01 \
  --gm_scale 0.95 \
  --verbosity "regular" \