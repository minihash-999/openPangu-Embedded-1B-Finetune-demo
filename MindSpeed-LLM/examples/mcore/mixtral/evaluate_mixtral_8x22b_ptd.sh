#!/bin/bash

# The number of parameters is not aligned
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
TP=8
PP=1

CHECKPOINT="Your ckpt file path"
TOKENIZER_PATH="Your vocab file path"
DATA_PATH="Your data path (such as ./mmlu/test/)"
TASK="mmlu"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size 1 \
    --moe-router-topk 2 \
"

GPT_ARGS="
    --use-mcore-models \
    --transformer-impl local \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --task $TASK \
    --task-data-path $DATA_PATH \
    --max-new-tokens 1 \
    --num-layers 56 \
    --hidden-size 6144 \
    --ffn-hidden-size 16384 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 65536 \
    --max-position-embeddings 65536 \
    --micro-batch-size 1 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --load ${CHECKPOINT}  \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --seed 42
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
    $GPT_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl | tee logs/evaluation_mcore_mixtral_8x22b_${TASK}.log
