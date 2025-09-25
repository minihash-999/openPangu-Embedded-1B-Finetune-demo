#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="Your ckpt file path"
TOKENIZER_PATH="Your tokenizer path"
DATA_PATH="./boolq/data/test/"
TASK="boolq"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py \
    --task-data-path ${DATA_PATH} \
    --task ${TASK} \
    --seq-length 4096 \
    --max-new-tokens 1 \
    --max-position-embeddings 4096 \
    --tensor-model-parallel-size 8  \
    --pipeline-model-parallel-size 1  \
    --num-layers 40  \
    --hidden-size 5120  \
    --ffn-hidden-size 13696 \
    --num-attention-heads 40  \
    --disable-bias-linear \
    --swiglu \
    --position-embedding-type alibi \
    --square-alibi-mask \
    --fill-neg-inf \
    --load $CHECKPOINT \
    --normalization RMSNorm \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --fp16 \
    --micro-batch-size 1  \
    --use-fused-rmsnorm \
    --exit-on-missing-checkpoint \
    --no-load-rng \
    --no-load-optim \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --make-vocab-size-divisible-by 32 \
    --use-mcore-models \
    --seed 42 | tee logs/eval_baichuan2_13b_mcore_${TASK}.log
