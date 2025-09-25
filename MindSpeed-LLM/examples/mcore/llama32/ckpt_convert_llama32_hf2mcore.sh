# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type-hf llama2 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 2 \
   --load-dir ./model_from_hf/llama32-hf/ \
   --save-dir ./model_weights/llama32-mcore/ \
   --tokenizer-model ./model_from_hf/llama32-hf/tokenizer.json