# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./pairwise_dataset

python ./preprocess_data.py \
        --input ./dataset/orca_rlhf.jsonl \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --tokenizer-name-or-path ./model_from_hf/Meta-Llama-3-8B-Instruct/ \
        --output-prefix ./pairwise_dataset/orca_rlhf_llama3 \
        --workers 4 \
        --log-interval 1000 \
        --handler-name AlpacaStylePairwiseHandler \
        --prompt-type llama3 \
        --map-keys '{"prompt":"question", "query":"", "system":"system"}'
