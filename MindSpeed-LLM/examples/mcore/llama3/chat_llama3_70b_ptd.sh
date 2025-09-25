#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 4  \
       --pipeline-model-parallel-size 1  \
       --use-mcore-models \
       --task chat \
       --hf-chat-template \
       --top-p 0.9 \
       --temperature 0.6 \
       --use-kv-cache \
       --use-flash-attn \
       --use-fused-swiglu \
       --use-fused-rmsnorm \
       --use-fused-rotary-pos-emb \
       --use-rotary-position-embeddings \
       --load ${CHECKPOINT}  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --num-layers 80 \
       --hidden-size 8192  \
       --ffn-hidden-size 28672 \
       --position-embedding-type rope \
       --rotary-base 500000 \
       --seq-length 8192 \
       --max-position-embeddings 8192 \
       --max-new-tokens 256 \
       --group-query-attention \
       --num-query-groups 8 \
       --micro-batch-size 1 \
       --num-attention-heads 64 \
       --swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-5 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 16032 \
       --bf16 \
       --seed 42 \
       | tee logs/chat_mcore_llama3_70b.log

