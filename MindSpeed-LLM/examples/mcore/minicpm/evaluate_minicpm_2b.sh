#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer model path"
CHECKPOINT="your model directory path"

# configure task and data path
DATA_PATH="./mmlu/test/"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# configure generation parameters
python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation.py   \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --seq-length 4096 \
       --max-new-tokens 1 \
       --evaluation-batch-size 1 \
       --max-position-embeddings 4096 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 40  \
       --hidden-size 2304  \
       --ffn-hidden-size 5760 \
       --num-attention-heads 36  \
       --swiglu \
       --disable-bias-linear \
       --load ${CHECKPOINT}  \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --fp16  \
       --micro-batch-size 1  \
       --use-fused-rmsnorm \
       --position-embedding-type rope \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --no-masked-softmax-fusion \
       --make-vocab-size-divisible-by 1 \
       --use-mcore-models \
       --scale-emb 12 \
       --dim-model-base 256 \
       --scale-depth 1.4 \
       --seed 42 | tee logs/evaluation_minicpm_2b_${TASK}.log



