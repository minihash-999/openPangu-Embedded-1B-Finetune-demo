#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1


NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model load ckpt path"

TP=8
PP=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 24576 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --load ${CKPT_LOAD_DIR} \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 8192 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --make-vocab-size-divisible-by 64 \
    --lr 1e-5 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-6 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --rotary-base 1000000 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --tokenizer-kwargs 'eos_token' '<|endoftext|>' 'pad_token' '<|extra_0|>' \
    --distributed-backend nccl \
    --jit-compile \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_qwen_72b.log
