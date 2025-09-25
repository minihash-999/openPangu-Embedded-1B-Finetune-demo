#!/bin/bash

# See README, please remember to source the set_env.sh file in CLI, or here
# source /path/to/your/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CKPT_SAVE_DIR="your checkpoint save dir"
DATA_PATH="your training data dir"
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

GPT_ARGS="
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --attention-dropout 0.0 \
    --attention-softmax-in-fp32 \
    --clip-grad 1.0 \
    --disable-bias-linear \
    --ffn-hidden-size 11008 \
    --fp16 \
    --global-batch-size 64 \
    --hidden-dropout 0.0 \
    --hidden-size 4096 \
    --init-method-std 0.01 \
    --initial-loss-scale 65536 \
    --lr 1.0e-7 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --make-vocab-size-divisible-by 1 \
    --max-position-embeddings 2048 \
    --micro-batch-size 8 \
    --min-lr 1.0e-8 \
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
    --sequence-parallel \
    --swiglu \
    --tensor-model-parallel-size ${TP} \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --tokenizer-type PretrainedFromHF \
    --train-iters 2000 \
    --untie-embeddings-and-output-weights \
    --use-flash-attn \
    --use-fused-rmsnorm \

    --weight-decay 1e-1 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 0
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load $CKPT_LOAD_DIR \
    | tee logs/train_aquila2_7b_ptd.log