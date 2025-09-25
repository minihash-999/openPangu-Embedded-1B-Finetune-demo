# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt.py \
    --model-type-hf gemma \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --params-dtype bf16 \
    --load-dir ./model_from_hf/gemma-hf \
    --save-dir ./model_weights/gemma-legacy \
    --tokenizer-model ./model_from_hf/gemma-hf/tokenizer.model