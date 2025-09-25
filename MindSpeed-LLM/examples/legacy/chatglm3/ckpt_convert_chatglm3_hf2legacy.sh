# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --model-type-hf chatglm3 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_from_hf/Chatglm3-hf/ \
    --save-dir ./model_weights/Chatglm3-legacy/ \
    --tokenizer-model ./model_from_hf/Chatglm3-hf/tokenizer.model \
    --add-qkv-bias