# 请按照您的真实环境修改 set_env.sh 路径
#source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --moe-grouped-gemm \
    --model-type-hf deepseek2-lite \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --load-dir ./model_weights/deepseek2_lite_mcore/ \
    --save-dir ./model/deepseek2_lite/
