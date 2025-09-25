#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=12
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model load ckpt path"



TP=8
PP=12

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
    --num-layers 70 \
    --hidden-size 14336 \
    --load ${CKPT_LOAD_DIR} \
    --num-attention-heads 112 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 2 \
    --global-batch-size 2048 \
    --num-layer-list 5,6,6,6,6,6,6,6,6,6,6,5 \
    --make-vocab-size-divisible-by 1 \
    --attention-softmax-in-fp32 \
    --apply-query-key-layer-scaling \
    --lr 1.2e-4 \
    --train-iters 5000 \
    --init-method-std 0.0048 \
    --hidden-dropout 0.0 \
    --position-embedding-type alibi \
    --normalization LayerNorm \
    --min-lr 6e-6 \
    --no-masked-softmax-fusion \
    --lr-decay-iters 200 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 70 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --seed 42
"

DATA_ARGS="
    --data-path $DATA_PATH
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1 \
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --jit-compile \
    --save $CKPT_SAVE_DIR \
    | tee logs/train_bloom_176b.log