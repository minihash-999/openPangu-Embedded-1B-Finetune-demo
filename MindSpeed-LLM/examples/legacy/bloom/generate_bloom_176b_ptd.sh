#!/bin/bash

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

export HCCL_CONNECT_TIMEOUT=1200

# please fill these path configurations
CHECKPOINT="your model save ckpt path"
TOKENIZER_PATH="your tokenizer path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=5
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 5 \
    --num-layers 70 \
    --embed-layernorm \
    --hidden-size 14336 \
    --padded-vocab-size 250880 \
    --load ${CHECKPOINT} \
    --num-attention-heads 112 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --make-vocab-size-divisible-by 1 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type alibi \
    --normalization LayerNorm \
    --attention-softmax-in-fp32 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    | tee logs/generate_bloom_176b.log
