# 多样本Pack微调

## Pack数据集处理方法

### 数据集

可以使用任意的`Alpaca`和`ShareGPT`的数据集。

### 数据集下载

`Alpaca`数据下载部分详见[**Alpaca风格数据的说明文档**](alpaca_dataset.md)。

`ShareGPT`数据下载部分详见[**ShareGPT风格数据的说明文档**](sharegpt_dataset.md)。

**接下来将以在使用LLaMA2-7B进行微调时，以Alpaca数据集作为输入进行多样本Pack微调进行说明。**

Pack数据集处理时，会将指定数据集中，不同长度的数据组合成指定长度，并尽可能地填充有效的数据内容。若拼接的数据无法达到指定的`seq-length`的长度，则该数据将会被Pad到指定的长度。因此，**每条Pack数据集的长度都一致**。

```bash
# 请根据 examples/README.md 下 “数据集准备及处理” 章节下载 Alpaca 数据集
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Llama2-hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --overwrite-cache \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2 \
    --pack \
    --append-eod \
    --seq-length 4096 \
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传
```

### 多样本Pack数据集相关参数

【--pack】

将数据转为Pack格式。

【--seq-length】

指定Pack数据集每条数据的长度。

【--append-eod】

在每个输入序列的末尾添加一个特殊的标记来表示输入序列的结束。

【--overwrite-cache】

用于控制是否覆盖已存在的缓存分词器。

【--input】

可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持 .parquet \ .csv \ .json \ .jsonl \ .txt \ .arrow 格式， 同一个文件夹下的数据格式需要保持一致 

【--map-keys】

`--map-keys`参数用于配置字段映射来使用数据集。

Alpaca风格示例请参考[**Alpaca风格数据的说明文档**](alpaca_dataset.md)。

### 启动脚本

MindSpeed-LLM微调Pack数据集处理脚本命名风格及启动方法为：

```shell
# mcore
# 命名及启动：examples/mcore/llama2/data_convert_llama2_instruction_pack.sh
bash examples/mcore/llama2/data_convert_llama2_instruction_pack.sh
```

指令微调数据集处理结果如下：

```shell
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```

微调时，数据集路径输入 `./finetune_dataset/alpaca` 即可

## 启动多样本Pack微调脚本

使用LLaMA2-7B模型目录下的<a href="../../examples/mcore/llama2/tune_llama2_7B_full_pack_ptd.sh">多样本Pack微调脚本</a>。

#### 填写相关路径

`DATA_PATH`：指定数据预处理后的保存路径。

`TOKENIZER_MODEL`：指定模型的分词器路径（例如`tokenizer.model`）。

`CKPT_LOAD_DIR`：指向权重转换后保存的路径。

`CKPT_SAVE_DIR`：指向用户指定的微调后权重保存路径（例如指定保存到`ckpt`文件夹下）。

因此，根据之前的示例，路径应填写如下：

```bash
DATA_PATH="./finetune_dataset/alpaca"
CKPT_SAVE_DIR="./ckpt/"
TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf/"
CKPT_LOAD_DIR="./model_weights/llama-2-7b-legacy/"
```

#### 多样本Pack微调相关参数

【--is-instruction-dataset】

用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。

【--prompt-type】

用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../mindspeed_llm/tasks/preprocess/templates.py)文件内查看。

【--reset-position-ids】

每条数据由不同的样本拼接而成，因此其位置 ID 并不连续。该参数用于为每条拼接的数据重置位置 ID，以确保在处理多个样本时，位置编码保持一致性。

#### 运行脚本

```bash
bash examples/mcore/llama2/tune_llama2_7B_full_pack_ptd.sh
```
