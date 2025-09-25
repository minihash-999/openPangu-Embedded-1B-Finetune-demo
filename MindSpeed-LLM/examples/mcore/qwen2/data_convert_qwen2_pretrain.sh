
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

python ./preprocess_data.py \
	--input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
	--tokenizer-name-or-path ./model_from_hf/qwen2_hf \
	--output-prefix ./dataset/alpaca \
	--tokenizer-type PretrainedFromHF \
	--workers 4 \
	--log-interval 1000


