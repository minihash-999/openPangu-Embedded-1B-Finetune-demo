[使用指南](./USER_GUIDE.md) | [开发指南](./DEVELOP_GUIDE.md)

<p align="left">
</p>

## 基于 Megatron-LM + MindSpeed-LLM 训练自定义大模型

本开发指南讲述基于 Megatron-LM 和 MindSpeed-LLM 来训练一个自定义的大模型。以下是具体步骤：

### 1. 获取 Megatron-LM
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.6.0
```

### 2. 获取 MindSpeed-Core 与 MindSpeed-LLM
```bash
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
pip install -r requirements.txt 
pip install -e .

git clone https://gitee.com/ascend/MindSpeed-LLM.git
cp -r MindSpeed-LLM/mindspeed_llm Megatron-LM
```

### 3. 快速上手

以 Megatron-LM GPT 预训练为例：

```bash
# 将 Megatron-LM 下的 pretrain_gpt.py 替换为 MindSpeed-LLM 下的 pretrain_gpt.py
cp MindSpeed-LLM/pretrain_gpt.py Megatron-LM/
# 将 MindSpeed-LLM 下的 preprocess_data.py 拷贝到 Megatron-LM
cp MindSpeed-LLM/preprocess_data.py Megatron-LM/
```

**(1) 转换预训练数据集**

将原始数据集转换成预训练支持格式：

```bash
python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
    --output-prefix ./dataset/gpt_text_sentence \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file ./vocab_file/gpt2-vocab.json \
    --merge-file ./vocab_file/gpt2-merges.txt \
    --append-eod \
    --workers 4 \
    --log-interval 1000
```

**(2) 配置脚本并启动训练**

在完成数据集转换后，配置词表路径：

```bash
CKPT_SAVE_DIR="your model save ckpt path"   # 模型训练好后权重保存地址
DATA_PATH="your data path"  # 数据集路径
TOKENIZER_MODEL="your tokenizer path"   # 分词器路径
```

根据 [Megatron-LM GPT模型 README](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/gpt3/README.md) 模型参数：
```bash
--num-layers 12 \
--hidden-size 512 \
--num-attention-heads 8 \
--seq-length 1024 \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \

--tokenizer-type GPT2BPETokenizer \
--no-gradient-accumulation-fusion \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion
```

拉起脚本：

```bash
bash examples/gpt3/train_gpt3_345M_distributed.sh
```

