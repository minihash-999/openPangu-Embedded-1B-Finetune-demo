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
# from tests.ut.inference.test_inference import acquire_context

PATTERN = r"MindSpeed-LLM:\n(.*)"


def acquire_context(log_capture):
    # Acquire the final score for evaluation tasks, still universal.
    context_str = log_capture[0]
    context_pattern = r"MindSpeed-LLM:\s*([\s\S]*)"
    match = re.search(context_pattern, context_str)
    if match:
        context = match.group(1)
    else:
        raise ValueError("No matching context found in the provided log.")
    return context


class TestInference(DistributedTest):
    world_size = 8
    cur_dir = Path(__file__).parent
    json_file = next(cur_dir.glob("*.json"), None)
    test_config = create_testconfig(json_file)

    @pytest.mark.parametrize("params", test_config["test_bloom_7B_greedy_search"])
    def test_greedy_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=============== bloom_7B greedy search =============")
            print(log_capture)
            context = acquire_context(log_capture)
            print(context)
            assert [context] == [
                '"\n\n"I\'m all right," said the boy, with a smile. "I was just thinking of\nyou."'
            ], [context]