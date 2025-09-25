# 修改 ascend-toolkit 路径
 source /usr/local/Ascend/ascend-toolkit/set_env.sh

 # 权重格式转换
 python convert_ckpt.py \
     --model-type GPT \
     --load-model-type hf \
     --save-model-type mg \
     --target-tensor-parallel-size 1 \
     --target-pipeline-parallel-size 1 \
     --add-qkv-bias \
     --load-dir ./model_from_hf/qwen15-hf/ \
     --save-dir ./model_weights/qwen15-legacy/ \
     --tokenizer-model ./model_from_hf/qwen15-hf/tokenizer.json \
     --model-type-hf llama2