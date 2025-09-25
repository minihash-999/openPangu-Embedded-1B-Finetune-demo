OUTPUT_DIR=/cache/finetune_dataset/
mkdir "$OUTPUT_DIR" -p
python ./preprocess_data.py \
    --input /cache/finetune_dataset/ \
    --output-prefix "${OUTPUT_DIR}/merge_demo_data" \
    --merge-group-keys packed_attention_mask_document packed_input_ids_document packed_labels_document