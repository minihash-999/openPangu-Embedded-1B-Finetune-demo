#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DATA_PATH="/data/bloom-7B-hf/bloom-7B-data/alpaca_text_document"
TOKENIZER_PATH="/data/bloom-7B-hf/"
CKPT_LOAD_DIR="/data/bloom-7B-hf/"

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 30 \
    --hidden-size 4096 \
    --load ${CKPT_LOAD_DIR} \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --embed-layernorm \
    --padded-vocab-size 250880 \
    --make-vocab-size-divisible-by 1 \
    --attention-softmax-in-fp32 \
    --apply-query-key-layer-scaling \
    --lr 1.2e-6 \
    --train-iters 20 \
    --init-method-std 0.0048 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --position-embedding-type alibi \
    --normalization LayerNorm \
    --min-lr 1e-8 \
    --lr-decay-iters 430000 \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --use-deter-comp \
    --seed 42
"

DATA_ARGS="
    --data-path $DATA_PATH
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 1 \
    --finetune \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl