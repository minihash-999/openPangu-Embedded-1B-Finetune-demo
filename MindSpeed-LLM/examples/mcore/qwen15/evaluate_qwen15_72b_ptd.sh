#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1800

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer path"
DATA_PATH="./mmlu/data/test"
TASK="mmlu"

# Change for multinode config
MASTER_ADDR=localhost
NPU_PER_NODE=8
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPU_PER_NODE*$NNODES))

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
    --use-mcore-models \
    --task-data-path $DATA_PATH \
    --task $TASK \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --ffn-hidden-size 24576 \
    --max-position-embeddings 8192 \
    --seq-length 8192 \
    --padded-vocab-size 152064 \
    --rotary-base 1000000 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --micro-batch-size 1 \
    --swiglu \
    --disable-bias-linear \
    --add-qkv-bias \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --load ${CHECKPOINT} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --exit-on-missing-checkpoint \
    --no-load-rng \
    --no-load-optim \
    --tokenizer-not-use-fast \
    --max-new-tokens 1 \
    --bf16 \
    --seed 42 \
    --no-chat-template \
    | tee logs/eval_mcore_qwen15_72b_${TASK}.log