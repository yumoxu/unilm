export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm
export SPLIT=val

export GOLD_PATH=${PROJ_PATH}/data/rouge_estimated-rr-34_config-25000_iter-masked_summ-mn-marge_cluster-ratio-reveal_0.0_prepend_len/${SPLIT}.json
export CKPT=*
export PRED_PATH="${PROJ_PATH}/model/unilm_13/ckpt-${CKPT}.${SPLIT}"
export python=$PROJ_PATH/bin/python
export python_file=$PROJ_PATH/s2s-ft/evaluations/eval_for_multinews.py

$python ${python_file} \
    --pred "${PRED_PATH}" \
    --gold ${GOLD_PATH} \
    --split ${SPLIT} \
    --save_best \
    --processes 1 \
    --trunc_len 300
