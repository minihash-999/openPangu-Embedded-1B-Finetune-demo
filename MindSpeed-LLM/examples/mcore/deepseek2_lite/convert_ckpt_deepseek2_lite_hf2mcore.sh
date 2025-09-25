# 请按照您的真实环境修改 set_env.sh 路径
# 按照您的实际需要修改目录信息并完成对应的TP、PP、EP的参数配置

#source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
   --use-mcore-models \
   --moe-grouped-gemm \
   --model-type-hf deepseek2-lite \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --target-expert-parallel-size 8 \
   --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
   --load-dir ./model_from_hf/deepseek_v2_lite/ \
   --save-dir ./model_weights/deepseek2_lite_mcore/ \
   --tokenizer-model ./model_from_hf/deepseek_v2_lite/

