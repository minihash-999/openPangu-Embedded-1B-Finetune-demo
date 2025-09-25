# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type-hf gemma \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/gemma_mcore/ \
    --save-dir ./model_from_hf/gemma_hf/  # 需要填入原始HF模型路径，新权重会存于./model_from_hf/gemma_hf/mg2hg/