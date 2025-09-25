#!/bin/bash

# See README, please remember to source the set_env.sh file in CLI, or here
# source /path/to/your/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CKPT_LOAD_DIR="your checkpoint load dir"
TOKENIZER_PATH="your tokenizer path"

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

torchrun $DISTRIBUTED_ARGS inference.py \
    --attention-softmax-in-fp32 \
    --bf16 \
    --disable-bias-linear \
    --exit-on-missing-checkpoint \
    --ffn-hidden-size 24576 \
    --group-query-attention \
    --hidden-size 6144 \
    --load $CKPT_LOAD_DIR \
    --make-vocab-size-divisible-by 1 \
    --max-new-tokens 512 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-masked-softmax-fusion \
    --norm-epsilon 1e-5 \
    --normalization RMSNorm \
    --num-attention-heads 48 \
    --num-layers 60 \
    --num-query-groups 8 \
    --pipeline-model-parallel-size $PP \
    --position-embedding-type rope \
    --seq-length 4096 \
    --swiglu \
    --tensor-model-parallel-size $TP \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --tokenizer-not-use-fast \
    --tokenizer-type PretrainedFromHF \
    --untie-embeddings-and-output-weights \
    --use-fused-rmsnorm \
    | tee logs/generate_aquila2_34b_ptd.log