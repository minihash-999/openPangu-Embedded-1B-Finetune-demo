# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf llama2 \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/Codellama-mcore/ \
    --save-dir ./model_from_hf/Codellama-hf/
