#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="Your ckpt file path"
TOKENIZER_PATH="Your tokenizer path"
DATA_PATH="./boolq/data/test/"
TASK="boolq"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation.py   \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --seq-length 2048 \
       --max-new-tokens 1 \
       --max-position-embeddings 2048 \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 80  \
       --hidden-size 8192  \
       --ffn-hidden-size 22016 \
       --num-attention-heads 64  \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load ${CHECKPOINT}  \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --fp16  \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --no-masked-softmax-fusion \
       --make-vocab-size-divisible-by 1 \
       --seed 42 | tee logs/eval_llama_65b_${TASK}.log