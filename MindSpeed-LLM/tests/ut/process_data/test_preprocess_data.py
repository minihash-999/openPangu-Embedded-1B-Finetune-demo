import os
from pathlib import Path
import pytest
import pandas as pd
from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import build_args, create_testconfig, compare_file_md5_same
from preprocess_data import main


class TestProcessPretrainData:
    """
        The pretrain dataset is divided into two parts, 
        individual processing results as well as results from the merge pretrain dataset.
        The three designed test cases are as follows: 
        1. processing of the first segment of the split pretrain dataset
        2. processing of the second segment of the split pretrain dataset
        3. merging the two segments and processing them together.
    """

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    
    @pytest.mark.parametrize("full_params, params, slice_range", 
    [
        (test_config["pretrain_dataset"][0], test_config["test_pretrain_datasets_part1"][0], slice(0, 25000)),
        (test_config["pretrain_dataset"][0], test_config["test_pretrain_datasets_part2"][0], slice(25000, None))
    ])
    def test_pretrain_datasets(self, build_args, full_params, params, slice_range):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-part"]):
            os.makedirs(full_params["test-out-part"])

        # read and split dataset
        df = pd.read_parquet(full_params["input-dataset"])
        df.iloc[slice_range, :].to_parquet(params["input"])

        # process pretrain datasets
        if slice_range == slice(0, 25000):
            print("=============== preprocess pretrain datasets part1 =============")
        elif slice_range == slice(25000, None):
            print("=============== preprocess pretrain datasets part2 =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        end_strs = ["_text_document.bin", "_text_document.idx"]
        for end_str in end_strs:
            test_file = params["output-prefix"] + end_str
            end_str = prefix_str + end_str
            base_file = full_params["base-out-part"] + end_str
            assert compare_file_md5_same(base_file, test_file)
        

    @pytest.mark.parametrize("full_params, params",
        [(test_config["pretrain_dataset"][0], test_config["test_merge_pretrain_datasets"][0])])
    def test_pretrain_merge_datasets(self, build_args, full_params, params):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-merge"]):
            os.makedirs(full_params["test-out-merge"])

        # merge pretrain dataset
        print("=============== merge pretrain datasets =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        end_strs = ["_text_document.bin", "_text_document.idx"]
        for end_str in end_strs:
            test_file = params["output-prefix"] + end_str
            end_str = prefix_str + end_str
            base_file = full_params["base-out-merge"] + end_str
            assert compare_file_md5_same(base_file, test_file)
            

class TestProcessInstructionData:
    """
        The instruction dataset is divided into two parts, 
        individual processing results as well as results from the merge instruction dataset.
        The three designed test cases are as follows: 
        1. processing of the first segment of the split instruction dataset
        2. processing of the second segment of the split instruction dataset
        3. merging the two segments and processing them together.
    """

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("full_params, params, merge_params, slice_range", 
        [
            (test_config["instruction_dataset"][0], test_config["test_instruction_datasets_part1"][0], test_config["test_merge_instrction_datasets"][0], slice(0, 25000)),
            (test_config["instruction_dataset"][0], test_config["test_instruction_datasets_part2"][0], test_config["test_merge_instrction_datasets"][0], slice(25000, None))
        ])
    def test_instruction_datasets(self, build_args, full_params, params, merge_params, slice_range):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-part"]):
            os.makedirs(full_params["test-out-part"])

        # read and split dataset
        df = pd.read_parquet(full_params["input-dataset"])
        df.iloc[slice_range, :].to_parquet(params["input"])

        # process instruction datasets
        if slice_range == slice(0, 25000):
            print("=============== preprocess instruction datasets part1 =============")
        elif slice_range == slice(25000, None):
            print("=============== preprocess instruction datasets part2 =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = [merge_params["merge-group-keys"][0], merge_params["merge-group-keys"][1], merge_params["merge-group-keys"][2]]
        end_suffixes = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixes:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-part"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)

    
    @pytest.mark.parametrize("full_params, params", 
        [(test_config["instruction_dataset"][0], test_config["test_merge_instrction_datasets"][0])])
    def test_merge_instruction_datasets(self, build_args, full_params, params):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-merge"]):
            os.makedirs(full_params["test-out-merge"])

        # merge instruction dataset
        print("=============== merge instruction datasets =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = [params["merge-group-keys"][0], params["merge-group-keys"][1], params["merge-group-keys"][2]]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-merge"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)
