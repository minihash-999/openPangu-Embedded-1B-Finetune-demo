# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset/internlm2-chat-20b/

python ./preprocess_data.py \
        --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
        --tokenizer-name-or-path ./model_from_hf/internlm2-chat-20b/ \
        --output-prefix ./dataset/internlm2-chat-20b/enwiki \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF