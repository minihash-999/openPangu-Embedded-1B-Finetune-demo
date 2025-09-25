# 后训练方法DPO(直接偏好对齐)

直接偏好优化（Direct Preference Optimization，DPO）是一种用于优化大型语言模型（LLMs）的训练方法，其目的是通过人类偏好数据来调整模型参数。与PPO等其他复杂的RLHF方法需要在训练过程中更新多个模型相比，DPO在训练过程中只需要对policy一个模型进行参数更新，训练效率更高。

![](../../sources/images/dpo/dpo_vs_rlhf.png)

这里实现的是经典DPO算法，在训练过程中，不会使用policy-model进行推理并参与其权重更新，因此是offline的。

## 使用说明

### 数据预处理

dpo使用pairwise正负样本配对数据集，数据预处理命令如下：

```shell
python ./preprocess_data.py \
    --input ./dataset/orca_rlhf.jsonl \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-not-use-fast \
    --tokenizer-name-or-path ./model_from_hf/Meta-Llama-3-8B-Instruct/ \
    --output-prefix ./pair_dataset/orca_rlhf_llama3 \
    --workers 4 \
    --log-interval 1000 \
    --handler-name AlpacaStylePairwiseHandler \
    --prompt-type llama3 \
    --map-keys '{"prompt":"question", "query":"", "system":"system"}'
```

更多关于pairwise数据集预处理说明见：[Pairwise-dataset](./pairwise_dataset.md ) 。

### 训练参数

dpo训练脚本参照：examples/mcore/llama3/dpo_llama3_8b_full_ptd.sh

相较于普通预训练，dpo需要增加一些几个参数：

- **`--stage`**

  必选，用于指定训练方法为dpo

- **`--is-pairwise-dataset`**

  必选，指定dpo使用pairwise配对数据集

- **`--dpo-loss-type `**

  可选参数，指定dpo计算loss方法，支持：sigmoid, hinge, ipo，默认sigmoid

- **`--dpo-beta`**

  可选参数，指定dpo计算loss中的参数，默认0.1

- **`--dpo-label-smoothing`**

  可选参数，dpo计算loss中的参数，取值范围0到0.5，默认0.0

- **`--pref-ftx`**

  可选参数，dpo计算loss中的参数，默认0.0

- **`--ref-model`**

  可选参数，指定参考模型路径， 默认为None，即与训练模型相同

注：当前版本DPO仅支持非MOE模型全参训练，分布式训练仅支持TP和PP切分。

## 参考文献

- [Direct Preference Optimization](https://export.arxiv.org/abs/2305.18290)