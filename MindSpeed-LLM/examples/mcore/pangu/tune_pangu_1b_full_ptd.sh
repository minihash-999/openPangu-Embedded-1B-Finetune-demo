#!/bin/bash
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
CKPT_SAVE_DIR="/cache/sft_ouputs/"
DATA_PATH="/cache/finetune_data/merge_demo_data"
TOKENIZER_MODEL="/cache/ckpts/pangu_model/"
CKPT_LOAD_DIR="/cache/ckpt/pangu_model_mcore"
SEQ_LENS=32768

TP=1
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
GPT_ARGS="
    --finetune \
    --stage sft \
    --use-mcore-models \

    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size 2 \
    --expert-model-parallel-size 1 \
    --sequence-parallel \
    
    --seed 1234 \
    --num-layers 26 \
    --hidden-size 1536 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 12 \
    --group-query-attention \
    --num-query-groups 6 \
    --kv-channels 128 \

    --tokenizer-type PretrainedFromHF \
    --tokenizer-not-use-fast \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LENS} \
    --max-position-embeddings ${SEQ_LENS} \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    
    --transformer-impl local \
    --distributed-timeout-minutes 120 \
    --make-vocab-size-divisible-by 16 \
    --padded-vocab-size 153376 \
    --lr-decay-style cosine \
    --lr 2e-5 \
    --min-lr 2e-6 \
    --lr-warmup-iters 200 \
    --override-opt_param-scheduler \

    --disable-bias-linear \
    --add-qkv-bias \
    --add-dense-bias \

    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --rotary-base 4000000 \ 
    
    --normalization RMSNorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \

    --optimizer adam \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --loss-scale 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \

    --initial-loss-scale 4096 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \

    --reset-attention-mask \
    --reset-position-ids \
    --eod-mask-loss \

    --is-instruction-dataset \
    --train-iters 600 \
    
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 26 \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --use-distributed-optimizer \
    
    --manual-gc \
    --manual-gc-interval 100 \
"
DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 0 \
"


torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee log.txt