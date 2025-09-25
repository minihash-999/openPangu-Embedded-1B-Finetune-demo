source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/Internlm-hf/ \
    --save-dir ./model_weights/Internlm-legacy/ \
    --tokenizer-model ./model_from_hf/Internlm-hf/tokenizer.model \
    --add-qkv-bias \
    --add-dense-bias \
    --model-type-hf llama2