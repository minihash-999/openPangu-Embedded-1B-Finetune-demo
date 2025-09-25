# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from mindspeed_llm import megatron_adaptor
from mindspeed_llm.tasks.post_train.launcher import AutoTrainer


def launch():
    trainer = AutoTrainer()
    trainer.train()


if __name__ == '__main__':
    launch()
