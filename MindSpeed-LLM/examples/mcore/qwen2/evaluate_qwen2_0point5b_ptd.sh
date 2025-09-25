#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6012
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CHECKPOINT="Your ckpt file path"
TOKENIZER_PATH="Your vocab file path"
DATA_PATH="Your data path (such as ./mmlu/test/)"
TASK="mmlu"

TP=1
PP=1
MBS=2
SEQ_LEN=4096

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
       --task-data-path ${DATA_PATH} \
       --task ${TASK} \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --micro-batch-size ${MBS} \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --max-new-tokens 1 \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 151936 \
       --num-layers 24  \
       --hidden-size 896  \
       --ffn-hidden-size 4864 \
       --num-attention-heads 14  \
       --group-query-attention \
       --num-query-groups 2 \
       --add-qkv-bias \
       --disable-bias-linear \
       --swiglu \
       --rotary-base 1000000 \
       --position-embedding-type rope \
       --load ${CHECKPOINT} \
       --normalization RMSNorm \
       --norm-epsilon 1e-6 \
       --tokenizer-not-use-fast \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --bf16 \
       --no-chat-template \
      | tee logs/eval_mcore_qwen2_0point5b_${TASK}.log
