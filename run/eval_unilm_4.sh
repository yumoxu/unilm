export PROJ_PATH=/disk/nfs/ostrom/s1617290/unilm
export SPLIT=val
export GOLD_PATH=${PROJ_PATH}/data/multinews-250/${SPLIT}.json
export CKPT=*
export PRED_PATH="${PROJ_PATH}/model/unilm_4/ckpt-*.${SPLIT}"
export python=$PROJ_PATH/bin/python
export python_file=$PROJ_PATH/s2s-ft/evaluations/eval_for_multinews.py

$python ${python_file} \
    --pred "${PRED_PATH}" \
    --gold ${GOLD_PATH} \
    --split ${SPLIT} \
    --save_best \
    --processes 32 \
    --trunc_len 300
