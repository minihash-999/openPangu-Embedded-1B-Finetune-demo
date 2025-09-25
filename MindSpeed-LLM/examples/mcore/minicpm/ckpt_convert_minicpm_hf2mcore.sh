# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换，设置需要的并行配置
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/MiniCPM-2B-sft-bf16/ \
    --save-dir ./model_from_hf/MiniCPM-2B-sft-mcore/ \
    --tokenizer-model ./model_from_hf/MiniCPM-2B-sft-bf16/tokenizer.model \
    --use-mcore-models \
    --model-type-hf minicpm