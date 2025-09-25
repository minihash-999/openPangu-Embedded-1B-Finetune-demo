# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ./dataset

python ./preprocess_data.py \
        --input ./dataset/orca_rlhf.jsonl \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --tokenizer-name-or-path ./model_from_hf/Llama2-hf/ \
        --output-prefix ./dataset/orca_rlhf \
        --workers 4 \
        --log-interval 1000 \
        --handler-name AlpacaStylePairwiseHandler \
        --prompt-type llama2 \
        --seq-length 4096 \
        --map-keys '{"prompt":"question", "query":"", "system":"system"}'
