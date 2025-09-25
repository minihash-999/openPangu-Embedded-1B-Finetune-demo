MindSpeed-LLM支持基于昇腾芯片采集profiling数据，以提供对模型运行情况的分析，主要API如下：


```bash
--profile                        # 打开profiling采集数据开关
--profile-step-start  5          # 指定开启采集数据的步骤
--profile-step-end 6             # 指定结束采集数据的步骤，实际采集步数为 end-start，不包含end
--profile-ranks 0 1 2 3 4        # 指定采集数据的卡号，默认为-1，表示采集所有rank的profiling数据，可以设置为 0 1 2 3 4 5 6 7 8 9 列表指定全局卡号
--profile-level level2           # 数据采集水平，level0, 1, 2, 级别越高采集信息越多，默认为level0
--profile-with-cpu               # 是否采集CPU数据，加入参数采集
--profile-with-stack             # 采集指令运行堆栈，加入参数采集
--profile-with-memory            # 是否采集内存，加入参数采集
--profile-record-shapes          # 是否采集计算shape，加入参数采集
--profile-save-path ./profile_dir    # profiling数据采集保存路径
```