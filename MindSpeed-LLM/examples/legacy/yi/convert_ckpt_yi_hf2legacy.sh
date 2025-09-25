# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 2 \
   --load-dir ./model_from_hf/Yi-hf/ \
   --save-dir ./model_weights/Yi-legacy/ \
   --tokenizer-model ./model_from_hf/Yi-hf/tokenizer.model \
   --params-dtype bf16 \
   --model-type-hf llama2