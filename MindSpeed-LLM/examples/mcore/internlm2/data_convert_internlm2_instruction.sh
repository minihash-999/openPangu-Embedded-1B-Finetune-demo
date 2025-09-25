# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset/internlm2-chat-20b/

python ./preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/internlm2-chat-20b/ \
        --output-prefix ./dataset/internlm2-chat-20b/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --handler-name AlpacaStyleInstructionHandler \
        --prompt-type chatml