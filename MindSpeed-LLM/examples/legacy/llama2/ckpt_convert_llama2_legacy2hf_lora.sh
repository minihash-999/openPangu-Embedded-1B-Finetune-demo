# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/llama-2-7b-hf-v0.1-tp8-pp1/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hg/