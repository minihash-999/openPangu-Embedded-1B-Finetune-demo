#!/bin/bash

# The number of parameters is not aligned
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

MASTER_ADDR=localhost
MASTER_PORT=6040
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
TP=1
PP=8

# modify script model path and tokenizer path
CHECKPOINT="./ckpt/mistral/"  #model path
TOKENIZER_PATH="./model_from_hf/mistral-7b-hf/"  #tokenizer path
DATA_PATH="./dataset/mmlu/test"
TASK="mmlu"

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
    --task $TASK \
    --task-data-path $DATA_PATH \
    --max-new-tokens 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --load ${CHECKPOINT}  \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --seed 42 \
    --rotary-base 1000000 \
    --use-mcore-models \
    --transformer-impl local \
    --prompt-type mistral \
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
    $GPT_ARGS \
    --distributed-backend nccl \
    | tee logs/evaluation_mcore_mistral_full_${TASK}.log
