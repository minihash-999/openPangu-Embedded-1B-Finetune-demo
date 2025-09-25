source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir /cache/ckpt/pangu_model/ \
    --save-dir /cache/ckpt/pangu_model_mcore/ \
    --tokenizer-model /cache/ckpt/pangu_open_model \
    --add-qkv-bias \
    --add-dense-bias \
    --params-dtype bf16 \
    --use-mcore-models