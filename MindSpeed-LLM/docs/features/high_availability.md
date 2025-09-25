## 昇腾高可用性

### 一、特性介绍

分布式优化器的思想是通过将优化器状态均匀地分布在数据并行组中来节省内存。基于该思想，MindIO设计了将数据并行组切分成两个副本数据并行组的方案，副本优化器将优化器状态均匀分布在副本数据并行组，实现优化器状态均有备份。结合华为自研的高可用框架，可实现以下功能：

#### 1.TTP临终遗言功能: 

在训练过程中发生故障后，校验优化器中间状态数据的完整性和一致性，生成一次临终Checkpoint数据，恢复训练时能够通过该CheckPoint恢复到故障前一刻的状态，减少故障造成的训练迭代损失。

#### 2.UCE Step级重计算功能： 

昇腾芯片支持NPU卡内存发生UCE故障（内存不可修复）的实时检测，检测到UCE故障后，基于优化器状态副本机制并完成故障卡的在线修复并继续训练，减少训练损失。


### 二、使用说明：

#### **0.使用约束**：

由于原理限制，为了保证故障发生后，有完整的优化器状态数据，需要在ptd切分时保障Data Parallel Size大于1，在使用MoE特性时还要求稠密层与稀疏层的Data Parallel Size均大于1，在使用长序列并行特性时还要求dp_cp_size大于1。

详见：[MindIO TTP 约束限制-昇腾社区](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc3/clusterscheduling/ref/mindiottp/mindiotft005.html)

#### 1.环境准备

MindIO的功能以whl包的形式提供

mindio_ttp下载地址：[MindIO TTP 下载软件包-昇腾社区](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc3/clusterscheduling/ref/mindioacp/mindioacp009.html)

由于本特性对片上内存占用会有一定增加，推荐千卡及以上的大规模集群使用本特性，减少故障引起的机时损失。

#### 2.启动脚本中添加启动参数：

`--enable-high-availability`  # 使能开启高可用功能的总开关，并使能TTP临终遗言功能，保存checkpoint时要求全局至少存在一份完整的优化器数据；

`--enable-hbmfault-repair` # 使能进行片上内存故障，Step级重计算功能的开关；本功能将在线进行worker级修复，修复时要求全局至少存在一个故障卡的副本卡。

`--enable-worker-reboot` # 使能空中加油功能，配合支持相关功能的 MindX DL 组件共同使能后，在发生一般性故障时，进行进程级重启修复，继续训练。本功能会将故障卡所在节点进行重启，修复时要求未故障节点中至少存在一份完整的优化器数据。

### 三、原理说明

megatron原生的分布式优化器数据流及工作原理如下图：

![输入图片说明](https://foruda.gitee.com/images/1720662898713744437/9f9003a6_14290444.png "屏幕截图")

副本优化器通过设计优化器参数均匀分布在副本DP组，完成优化器状态的备份，从而为TTP/UCE功能提供机制支持：

![输入图片说明](https://foruda.gitee.com/images/1720665330348419594/4aa04703_14290444.png "屏幕截图")

副本优化器相比分布式优化器会有内存占用增加，相对占用如下：

|                                  | Non-distributed optim | Distributed optim | Replica optim |
|----------------------------------|-----------------------|-------------------|---------------|
| fp16/bf16 param, fp16/bf16 grads | 20                    | 4 + 16/d          | 4 + 32/d      |
| fp16/bf16 param, fp32 grads      | 18                    | 6 + 12/d          | 6 + 24/d      |
| fp32 param, fp32 grads           | 16                    | 8 + 8/d           | 8 + 16/d      |



### 四、MindIO TTP 项目介绍

MindIO TTP 架构

![输入图片说明](https://foruda.gitee.com/images/1720665374547748248/3224f998_14290444.png "屏幕截图")

MindIO TTP 项目官网地址：
[MindIO TTP 项目介绍-昇腾社区](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc3/clusterscheduling/ref/mindiottp/mindiotft001.html)