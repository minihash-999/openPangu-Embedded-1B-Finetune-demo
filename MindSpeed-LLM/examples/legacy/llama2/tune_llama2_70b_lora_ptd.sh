#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save lora ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"
LORA_CHECKPOINT="your lora ckpt path"

TP=1
PP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# 断点续训需要添加参数 --lora-load ${LORA_CHECKPOINT} \

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --variable-seq-lengths \
    --load ${CKPT_LOAD_DIR} \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --tokenizer-not-use-fast \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 128  \
    --make-vocab-size-divisible-by 1 \
    --train-iters 1000 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --untie-embeddings-and-output-weights \
    --normalization RMSNorm \
    --swiglu \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --min-lr 1.0e-6 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --init-method-std 0.01 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 8 \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --rotary-base 10000 \
    --norm-epsilon 1e-05 \
    --vocab-size 32000 \
    --log-throughput \
    --prompt-type llama2 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee logs/tune_llama2_70b.log