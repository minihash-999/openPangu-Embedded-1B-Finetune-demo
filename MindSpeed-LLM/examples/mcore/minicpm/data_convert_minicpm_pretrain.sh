source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

python ./preprocess_data.py \
  --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
  --tokenizer-name-or-path ./model_from_hf/MiniCPM-2B-sft-bf16/ \
  --tokenizer-type PretrainedFromHF \
  --handler-name GeneralPretrainHandler \
  --output-prefix ./dataset/enwiki \
  --json-keys text \
  --workers 4 \
  --log-interval 1000