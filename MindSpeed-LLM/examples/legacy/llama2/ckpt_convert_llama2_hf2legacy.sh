# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换，设置需要的并行配置，--num-layers-per-virtual-pipeline-stage 5，--params-dtype bf16 结合需要使用
python convert_ckpt.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/llama-2-hf/ \
    --save-dir ./model_weights/llama-2-legacy/ \
    --tokenizer-model ./model_from_hf/llama-2-hf/tokenizer.model
