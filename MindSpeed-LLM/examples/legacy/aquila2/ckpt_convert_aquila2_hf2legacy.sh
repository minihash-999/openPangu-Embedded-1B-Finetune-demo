# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

#选择你需要的并行策略，--params-dtype bf16 结合需要选择使用
python convert_ckpt.py \
    --model-type GPT \
    --load-dir ./model_from_hf/Aquila2-hf/ \
    --save-dir ./model_weights/Aquila2-legacy/ \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --tokenizer-model ./model_from_hf/Aquila2-hf/tokenizer.json \
    --model-type-hf llama2
