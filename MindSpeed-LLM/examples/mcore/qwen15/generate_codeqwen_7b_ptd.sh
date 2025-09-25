#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 4  \
       --pipeline-model-parallel-size 2  \
       --use-mcore-models \
       --load ${CHECKPOINT}  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --num-layers 32 \
       --hidden-size 4096  \
       --ffn-hidden-size 13440 \
       --position-embedding-type rope \
       --rotary-base 1000000 \
       --seq-length 8192 \
       --max-position-embeddings 8192 \
       --max-new-tokens 256 \
       --group-query-attention \
       --num-query-groups 4 \
       --add-qkv-bias \
       --micro-batch-size 1 \
       --num-attention-heads 32  \
       --swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-5 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 92416 \
       --bf16 \
       --seed 42 \
       | tee logs/generate_mcore_codeqwen_7b.log
