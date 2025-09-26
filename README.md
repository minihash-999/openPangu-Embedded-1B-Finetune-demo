# openPangu-Embedded-1B-Finetune-Demo
最近基于华为开源的openPangu-Embedded-1B模型进行了续训，总结了一份训练指南，主要包括这几块：
1. 环境搭建。
2. 训练代码的修改，详见（续训代码部分）。
3. 数据处理的修改，为了最大程度保留模型的效果，需要将续训的数据格式处理为对应的格式。
4. 模型格式的转换，开源的格式为hf格式，在ascend训练需要mg格式，故在开始训练前需先完成模型格式的转换（hf->mg）。
5. 启动训练。
### 一、环境搭建
pytorch==2.1.0 cann==8.0.rc3 python==3.9.10

mindspeed-LLM使用1.0.0 
https://gitee.com/ascend/MindSpeed-LLM/tree/1.0.0 具体安装详情参考链接

注意： 使用transformers==4.53.2

### 二、续训代码
训练代码参考路径：https://gitee.com/ascend/Mindspeed-LLM

基于这套代码做了以下修改：

1、--add-dense-bias 参数有问题，没有生效（该参数是增加attention层的proj层的bias）

文件位置：megatron/core/transformer/attention.py

在attention类中的__init__方法下，修改self.linear_proj如下
```python
# Output
self.linear_proj = build_module(
    submodules.linear_proj,
    self.query_projection_size,
    self.config.hidden_size,
    config=self.configl,
    init_method=self.config.output_layer_init_method,
    # bias=self.config.add_bias_linear,
    bais=self.config.add_dense_bias or self.config.add_bias_linear,
    input_is_parallel=True,
    skip_bias_add=True,
    is_expert=False,
    tp_comm_buffer_name="proj",
)
```
2、增加transformer_config中的add_dense_bias参数

文件位置：megatron/core/transformer/transformer_config.py

在类TransformerConfig中增加如下参数
```python
add_dense_bias: bool = False
```

### 三、数据处理（cache和merge）
数据处理主要包括两部，数据cache和数据merge，根据盘古模型的回答格式，对其训练数据的格式进行了复原，因此增加了数据处理相关代码

文件位置：mindspeed_llm/tasks/preprocess/data_handler.py

在BaseDatasetHandler基类下面增加如下代码
```python
from dataclasses import dataclass
@dataclass
class PanguSftTemplate:
    system_token = "系统："
    user_token = "用户："
    assistant_token = "助手："
    tool_token = "工具："
    start_token = "[unused9]"
    end_token = "[unused10]"

class PanguPrompter(object):
    def __init__(self, template, verbose: bool=False):
        self._verbose = verbose
        self.template  = template
        self.user_role = "user"
        self.tool_role = "tool"
        self.assistant_role = "assistant"

class PanguInstructionHandler(BaseDatasetHandler):
    '''
    a general instruction dataset handler
    '''
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = PanguPrompter(PanguSftTemplate())
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        self.args.output_prefix += "_packed"
        self.ignored_label = -100
        self.is_multi_turn = True
        print("length of tokenizer is .", len(self.tokenizer.tokenizer))
    
    def _format_msg(self, data):
        messages = []
        turns = int(len(data) / 2)
        for i in range(turns):
            messages.append(data[i*2])
            messages.append(data[i*2+1])
        return messages

    def _pack_serialize_to_disk(self):
        '''save idx and bin to disk'''
        startup_start = time.time()
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.get_tokenized_data()
        output_bin_files, output_idx_files, builders = {}, {}, {}
        level = "document"
        if self.args.split_sentences:
            level = "sentence"
        logger.info("Vocal size: %s", self.tokenizer.vocab_size)
        logger.info("Output prefix: %s", self.args.output_prefix)
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key])
        
        self.output_idx_files = output_idx_files
        startup_end = time.time()
        proc_start = time.time()
        logger.info("Time to startup:%s", startup_end - startup_start)
        
        valid_num = 0
        key_data_dict = {key: [] for key in self.args.json_keys}
        lengths = []
        from collections import defaultdict
        length2indexes = defaultdict(list)
        # add lyl
        for _, doc in enumerate(iter(self.tokenized_dataset), start=1):
            batch = doc["input_ids"]
            label = doc.get("labels", None)

            for indice, sample in enumerate(batch):
                length = len(sample)
                if length > self.args.seq_length:
                    logger.warning(f"Dropped lengthy example with length {length} > {self.args.seq_length}.")
                else:
                    lengths.append(length)
                    length2indexes[length].append(valid_num)
                    for key in self.args.json_keys:
                        if key != "labels":
                            key_data_dict[key].append(sample)
                        else:
                            if not label:
                                key_data_dict[key].append(sample)
                            else:
                                key_data_dict[key].append(label[indice])
                    valid_num += 1
                
        logger.info(f"valid_num = {valid_num}. total_num = {len(self.tokenized_dataset)}, "
                    f"percentage: {valid_num / len(self.tokenized_dataset) * 100}%")

        knapsacks = greedy_knapsack(lengths, self.args.seq_length)
        for k, knapsack in enumerate(knapsacks):
            packed_data_dict = {key: [] for key in self.args.json_keys}

            for _, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                for key in self.args.json_keys:
                    packed_data_dict[key] += key_data_dict[key][index]
            
            if k % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                logger.info("Processed %s documents (%s docs/s). ", k, self.args.log_interval / elapsed)
            
            pad_length = self.args.seq_length - len(packed_data_dict["input_ids"])
            pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0 
            packed_data_dict["input_ids"] += [pad_token_id] * pad_length
            packed_data_dict["attention_mask"] += [1] * pad_length
            packed_data_dict["labels"] += [self.ignored_label] * pad_length
            
            for key in self.args.json_keys:
                if len(packed_data_dict[key]) != self.args.seq_length:
                    raise ValueError("The length of packed example should be indentical to the seq_length. ")
                sentence = torch.IntTensor(packed_data_dict[key])
                builders[key].add_item(sentence)
                builders[key].end_document()
        
        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def _filter(self, sample):
        messages = self._format_msg(sample["data"])
        tokenized_full_prompt = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
        full_prompt = ''
            # add sys token
        meta_prompts = sample['meta_prompt']
        if not isinstance(meta_prompts, list):
            meta_prompts = [meta_prompts]
            
        prompt = ''
        for meta in meta_prompts:
            prompt += self.prompter.template.start_token + self.prompter.template.system_token + meta + self.prompter.template.end_token
        tokenized_prompt = self.tokenizer.tokenizer.encode(prompt, add_special_tokens=True)
        tokenized_full_prompt["input_ids"].extend(tokenized_prompt)
        tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_prompt))
        full_prompt += prompt
        # add dialog token
        for message in messages:
            # usr/tool
            if message["role"] == self.prompter.user_role or message["role"] == self.prompter.tool_role:
                if message["role"] == self.prompter.user_role:
                    replaced_name = self.prompter.template.user_token
                else:
                    replaced_name = self.prompter.template.tool_token
                    
                prompt = self.prompter.template.start_token + replaced_name + message["content"] + self.prompter.template.end_token
                tokenized_prompt = self.tokenizer.tokenizer.encode(prompt, add_special_tokens=False)
                tokenized_full_prompt["input_ids"].extend(tokenized_prompt)
                tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_prompt))
                full_prompt += prompt
            else:
                answer = self.prompter.template.start_token + self.prompter.template.assistant_token
                tokenized_answer = self.tokenizer.tokenizer.encode(answer, add_special_tokens=False)
                tokenized_full_prompt["input_ids"].extend(tokenized_answer)
                tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_answer))
                full_prompt += answer

                answer = message["content"] + self.prompter.template.end_token
                tokenized_answer = self.tokenizer.tokenizer.encode(answer, add_special_tokens=False)
                tokenized_full_prompt["input_ids"].extend(tokenized_answer)
                tokenized_full_prompt["labels"].extend(tokenized_answer)
                full_prompt += answer
        tokenized_full_prompt["attention_mask"] = [1] * len(tokenized_full_prompt["input_ids"])
        # add eod
        if self.args.append_eod:
            print("eod_id is", self.tokenizer.eod)
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.ignored_label)
            
        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
            
        assert len(tokenized_full_prompt["input_ids"]) == len(tokenized_full_prompt["attention_mask"])
        assert len(tokenized_full_prompt["attention_mask"]) == len(tokenized_full_prompt["labels"])
        return tokenized_full_prompt
```

增加特性：

1、在每个数据加入system_prompt， 同时忽略其对应的loss计算（体现在labels的设置上）.

2、为每个数据中的对话开头和结束加上pangu射频的start_token和end_token字符.

3、为每个数据的开头加上一个代表开始的字符\<s\>.


#### 下面以开源数据集进行展示

##### 1、数据cache

下面是jsonl文件中的一个例子

{"meta_prompt": [""], "data": [{"role": "user", "content": "写一篇小红书风格的帖子，标题是男士护肤系列使用心得"}, {"role": "assistant", "content": "...."} ]}

针对原始数据集要处理成以上格式；只要包括"meta_prompt"字段和"data"字段

代码目录：./preprocess_data.py
```lua
mkdir /cache/finetune_dataset/
python ./preprocess_data.py \
    --input /cache/dataset/demo_data/ \
    --tokenizer-name-or-path /cache/ckpts/pangu_model/tokenizer \
    --output-prefix /cache/finetune_dataset/demo_data \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name PanguInstructionHandler \
    --seq-length 32768 \
    --pack

# 参数说明
# --input 数据集路径
# --tokenizer-name-or-path tokenizer路径
# --output-prefix 输出文件路径 + 文件开头
# --workers worker数目
# --log-interval 日志记录间隔
# --tokenizer-type PretrainedFromHF \
# --handler-name PanguInstructionHandler 盘古适配handler
# --seq-length 数据pack最长长度
# --pack 开启数据pack
```

注意：seq-length设置一定要和后面的sft训练时的seq-length相同！！！！！！！！！！！！！！！！

##### 2、数据merge

对于cache过程中，input下面的jsonl文件数目不为1的时候需要进行merge，反之不需要执行

代码目录：./preprocess_data.py
```lua
mkdir /cache/finetune_dataset/
python ./preprocess_data.py \
    --input /cache/finetune_dataset/ \
    --output-prefix /cache/finetune_dataset/merge_demo_data \
    --merge-group-keys packed_attention_mask_document packed_input_ids_document packed_labels_document
# 参数说明
# --input 数据集路径
# --output-prefix 输出文件路径 + 文件开头
# --merge-group-keys sft使用上述设置
```

### 四、模型转化
hf转mcore（后续训练使用）

启动脚本：examples/mcore/pangu/convert_hf2mcore.sh
```lua
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir /cache/ckpt/pangu_model/ \
    --save-dir /cache/ckpt/pangu_model_mcore/ \
    --tokenizer-model /cache/ckpt/pangu_open_model \
    --add-qkv-bias \
    --add-dense-bias \
    --params-dtype bf16 \
    --use-mcore-models
```

### 五、SFT启动
启动脚本：examples/mcore/pangu/fune_pangu_1B_full_ptd.sh
```lua
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

```
