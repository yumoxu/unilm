export MODEL_NAME=unilm_11

export PROJ_ROOT=/disk/nfs/ostrom/s1617290/unilm

export DATA_DIR=${PROJ_ROOT}/data/multinews-gold_rank_f1_0.15
export TRAIN_FILE=${DATA_DIR}/train.json

export OUTPUT_DIR=${PROJ_ROOT}/model/${MODEL_NAME}
export MODEL_DIR=${PROJ_ROOT}/model/unilm1.2-base-uncased
export CACHE_DIR=${PROJ_ROOT}/model/cache_unilm

export python=${PROJ_ROOT}/bin/python
export python_file=${PROJ_ROOT}/s2s-ft/run_seq2seq.py

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
$python -m torch.distributed.launch --nproc_per_node=8 ${python_file} \
  --train_file $TRAIN_FILE \
  --output_dir $OUTPUT_DIR \
  --model_type unilm \
  --model_name_or_path $MODEL_DIR \
  --do_lower_case \
  --fp16 \
  --fp16_opt_level O2 \
  --max_source_seq_length 768 \
  --max_target_seq_length 400 \
  --per_gpu_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 7e-5 \
  --num_warmup_steps 1000 \
  --num_training_steps 20000 \
  --cache_dir $CACHE_DIR \
  --save_steps 1500
