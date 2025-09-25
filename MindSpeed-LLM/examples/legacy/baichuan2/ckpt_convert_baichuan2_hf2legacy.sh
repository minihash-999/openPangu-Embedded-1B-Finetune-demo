# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置你需要的并行策略
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --load-dir ./model_from_hf/Baichuan2-hf/ \
    --save-dir ./model_weights/Baichuan2-legacy/ \
    --tokenizer-model ./model_from_hf/Baichuan2-hf/tokenizer.model \
    --params-dtype bf16 \
    --w-pack True \
    --model-type-hf baichuan2