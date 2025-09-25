#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISITIC=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
LOCAL_HOST=$(ifconfig enp189s0f0 | grep 'inet ' | awk '{print $2}')
MASTER_ADDR=$LOCAL_HOST
export MASTER_ADDR=$LOCAL_HOST
MASTER_PORT=6079
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_SAVE_DIR=/data/high_availability
CKPT_LOAD_DIR=/data/high_availability
DATA_PATH=/data/pretrain_dataset/alpaca_text_document
TOKENIZER_MODEL=/data/llama-2-7b-hf/tokenizer.model
TP=2
PP=1
rm -rf $CKPT_SAVE_DIR
pip install /home/high_availability/1031/mindio_ttp-1.0.0-cp38-cp38-linux_aarch64.whl --force-reinstall
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

DIST_ALGO=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --sequence-parallel
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 8
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 4096
)

TRAINING_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --micro-batch-size 1
    --global-batch-size 32
    --make-vocab-size-divisible-by 1
    --lr 1.25e-6
    --train-iters 15
    --lr-decay-style cosine
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --init-method-std 0.01
    --hidden-dropout 0.0
    --position-embedding-type rope
    --normalization RMSNorm
    --use-fused-rmsnorm
    --swiglu
    --use-flash-attn
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --min-lr 1.25e-7
    --weight-decay 1e-1
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --adam-beta1 0.9
    --initial-loss-scale 65536
    --adam-beta2 0.95
    --no-gradient-accumulation-fusion
    --no-load-optim
    --no-load-rng
    --use-fused-swiglu
    --use-fused-rotary-pos-emb
    --overlap-grad-reduce
    --bf16
    --enable-high-availability
    --enable-hbmfault-repair
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
)

OUTPUT_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 0
)

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/pretrain_gpt.py \
    ${DIST_ALGO[@]} \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --log-throughput \
    --distributed-backend nccl
