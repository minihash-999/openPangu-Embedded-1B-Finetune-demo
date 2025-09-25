import sys
import os
from pathlib import Path
import re
import logging
from torch import distributed as dist
import pytest
from inference import main
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import build_args, create_testconfig, setup_logger
from tests.ut.inference.test_inference import acquire_context


PATTERN = r"MindSpeed-LLM:\n(.*)"


class TestInference(DistributedTest):
    world_size = 8
    cur_dir = Path(__file__).parent
    json_file = next(cur_dir.glob("*.json"), None)
    test_config = create_testconfig(json_file)

    
    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_greedy_search"])
    def test_greedy_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=============== qwen15_7B greedy search =============")
            print(log_capture)
            context = acquire_context(log_capture)
            assert [context] == [
                "I'm sorry, but as an AI language model, I don't have feelings or emotions like humans do. I'm here to assist you with any questions or tasks you may have. How can I help you today?"
            ], [context]
