OUTPUT_DIR=/cache/finetune_dataset/
mkdir "$OUTPUT_DIR" -p
python ./preprocess_data.py \
    --input /cache/dataset/demo_data/ \
    --tokenizer-name-or-path /cache/ckpts/pangu_model/tokenizer \
    --output-prefix "${OUTPUT_DIR}/demo_data" \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name PanguInstructionHandler \
    --seq-length 32768 \
    --pack