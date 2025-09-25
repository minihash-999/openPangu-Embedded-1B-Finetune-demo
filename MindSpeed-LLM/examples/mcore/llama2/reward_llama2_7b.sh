#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=4
PP=2
MBS=1
GBS=8
SEQ_LEN=2048

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --max-position-embeddings 4096 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-7 \
    --lr-decay-style constant \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.2 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --clip-grad 1.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr-warmup-fraction 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --load-checkpoint-loosely \
    --use-distributed-optimizer \
    --weight-decay 0.0 \
    --is-pairwise-dataset \
    --finetune \
    --stage rm \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 9798,200,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100000 \
    --eval-interval 500 \
    --eval-iters 200 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --variable-seq-lengths \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    | tee logs/reward_llama2_7b.log
