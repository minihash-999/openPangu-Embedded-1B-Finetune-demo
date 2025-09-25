# 请按照您的真实环境修改 set_env.sh 路径
 source /usr/local/Ascend/ascend-toolkit/set_env.sh

 python convert_ckpt.py \
     --model-type GPT \
     --load-model-type mg \
     --save-model-type hf \
     --model-type-hf llama2 \
     --load-dir ./model_weights/qwen15-legacy/ \
     --target-tensor-parallel-size 1 \
     --target-pipeline-parallel-size 1 \
     --add-qkv-bias \
     --save-dir ./model_from_hf/qwen15-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/qwen15-hf/mg2hg/