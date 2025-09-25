#!/bin/bash

# A system test mainly for context parallel as well as tensor parallel and distributed optimizer.
# Also, several overlap algorithm have applied such as overlap-grad-reduce, overlap-param-gather
# and cp-send-recv-overlap.
# In addition, several fused kernels have applied, such as Rope, Swiglu and RmsNorm.
# Finally, re-computation with recompute-activation-function and recompute-activation-function-num-layers has tested.


export CUDA_DEVICE_MAX_CONNECTIONS=1

# config for multinode
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)


CKPT_SAVE_DIR=/data/ckpt
CKPT_LOAD_DIR=/data/llama-2-mcore-tp2-cp4-test
DATA_PATH=/data/pretrain_dataset/alpaca_text_document
TOKENIZER_MODEL=/data/llama-2-7b-hf/tokenizer.model

NUM_LAYER=8
GBS=16
TP=2
PP=1
CP=4
CP_ALGO=hybrid_cp_algo

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_ALGO} \
    --ulysses-degree-in-cp 2 \
    --use-cp-send-recv-overlap \
    --sequence-parallel \
    --kv-head-repeat-before-uly-alltoall \
"

ACCELERATE_ARGS="
    --reuse-fp32-param \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --recompute-activation-function \
    --recompute-activation-function-num-layers 1 \
"

MODEL_ARGS="
    --use-mcore-models \
    --transformer-impl local \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --num-layers ${NUM_LAYER} \
    --make-vocab-size-divisible-by 1 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --group-query-attention \
    --num-query-groups 4 \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
"

TRAIN_ARGS="
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --global-batch-size ${GBS} \
    --log-throughput \
    --lr 1.0e-6 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --no-gradient-accumulation-fusion \
    --no-save-optim
    --no-save-rng
    --no-load-optim
    --no-load-rng
    --disable-bias-linear \
    --bf16
"

FUSED_OP="
      --use-flash-attn \
      --use-fused-swiglu \
      --use-fused-rotary-pos-emb \
      --use-fused-rmsnorm
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

# add --finetune arguments here to solve checkpoint problem
OUTPUT_ARGS="
    --log-interval 1 \
    --train-iters 15 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 0 \
    --finetune
"


torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $PARALLEL_ARGS \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $FUSED_OP \
    $ACCELERATE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR
