#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"
DATA_PATH="./dataset/human_eval"
TASK="human_eval"

TP=4
PP=2
MBS=1
SEQ_LEN=65536

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py \
       --use-mcore-models \
       --task-data-path $DATA_PATH \
       --task ${TASK} \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --micro-batch-size ${MBS}  \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --max-new-tokens 1024 \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 92416 \
       --rotary-base 1000000 \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 13440 \
       --num-attention-heads 32 \
       --group-query-attention \
       --num-query-groups 4 \
       --add-qkv-bias \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load $CHECKPOINT \
       --normalization RMSNorm \
       --norm-epsilon 1e-06 \
       --tokenizer-not-use-fast \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --no-masked-softmax-fusion \
       --instruction-template "{prompt}" \
       | tee logs/eval_mcore_codeqwen_7b_${TASK}.log
