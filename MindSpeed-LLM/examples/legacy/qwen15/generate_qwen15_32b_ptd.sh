#!/bin/bash

# The number of parameters is not aligned
export  LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1800
export COMBINED_ENBLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations

CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer path"

# Change for multinode config
MASTER_ADDR=localhost
NPU_PER_NODE=8
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPU_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPU_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK \
                 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --num-layers 64 \
       --hidden-size 5120 \
       --num-attention-heads 40 \
       --ffn-hidden-size 27392 \
       --max-position-embeddings 8192 \
       --seq-length 8192 \
       --padded-vocab-size 152064 \
       --rotary-base 1000000 \
       --make-vocab-size-divisible-by 1 \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
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
       --norm-epsilon 1e-6 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --group-query-attention \
       --num-query-groups 8 \
       --tokenizer-not-use-fast \
       --max-new-tokens 256 \
       --bf16 \
       --seed 42 | tee logs/generate_qwen15_32b.log