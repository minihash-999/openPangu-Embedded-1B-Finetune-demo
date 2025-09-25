# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --load-dir ./model_from_hf/Llama3-hf/ \
   --save-dir ./model_weights/Llama3-legacy/ \
   --tokenizer-model ./model_from_hf/Llama3-hf/tokenizer.json \
   --model-type-hf llama2