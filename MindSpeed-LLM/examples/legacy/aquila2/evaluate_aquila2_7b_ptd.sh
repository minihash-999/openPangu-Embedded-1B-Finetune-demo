#!/bin/bash

# See README, please remember to source the set_env.sh file in CLI, or here
# source /path/to/your/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CKPT_LOAD_DIR="your checkpoint load dir"
TOKENIZER_PATH="your tokenizer path"
EVAL_DATA_PATH="your eval data dir"
TASK="your task name"

# Change for multinode config
TP=8
PP=1
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py \
    --attention-softmax-in-fp32 \
    --disable-bias-linear \
    --exit-on-missing-checkpoint \
    --ffn-hidden-size 11008 \
    --fp16 \
    --hidden-size 4096 \
    --load $CKPT_LOAD_DIR \
    --make-vocab-size-divisible-by 1 \
    --max-new-tokens 1 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-masked-softmax-fusion \
    --norm-epsilon 1e-5 \
    --normalization RMSNorm \
    --num-attention-heads 32 \
    --num-layers 32 \
    --pipeline-model-parallel-size ${PP} \
    --position-embedding-type rope \
    --seq-length 2048 \
    --swiglu \
    --task $TASK\
    --task-data-path $EVAL_DATA_PATH \
    --tensor-model-parallel-size ${TP} \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --tokenizer-not-use-fast \
    --tokenizer-type PretrainedFromHF \
    --untie-embeddings-and-output-weights \
    --use-fused-rmsnorm \
    | tee logs/eval_aquila2_7b_${TASK}_ptd.log