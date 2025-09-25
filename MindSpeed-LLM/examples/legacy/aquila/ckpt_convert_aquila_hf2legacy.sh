# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

#选择你需要的并行策略
python convert_ckpt.py \
    --model-type GPT \
    --load-dir ./model_from_hf/Aquila-hf/ \
    --save-dir ./model_weights/Aquila-legacy/ \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --tokenizer-model ./model_from_hf/Aquila-hf/tokenizer.json \
    --model-type-hf llama2