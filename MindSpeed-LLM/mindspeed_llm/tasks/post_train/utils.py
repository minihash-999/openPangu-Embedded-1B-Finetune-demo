# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from functools import wraps
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from mindspeed_llm.tasks.preprocess.decoder_packed_mtf_dataset import build_train_valid_test_datasets as build_instruction_dataset


def is_dataset_built_on_rank():
    return mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=args.mock_data,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if args.is_instruction_dataset or args.is_pairwise_dataset:
        train_ds, valid_ds, test_ds = build_instruction_dataset(
            data_prefix=args.data_path,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed)
    else:
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            config
        ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_attr_from_wrapped_model(model, target_attr):
    def recursive_search(module):
        if hasattr(module, target_attr):
            return getattr(module, target_attr)

        for _, child in getattr(module, '_modules').items():
            result = recursive_search(child)
            if result is not None:
                return result

        return None

    return [recursive_search(model)]


def get_tensor_shapes_decorator(get_tensor_shapes):
    @wraps(get_tensor_shapes)
    def wrapper(
            rank,
            model_type,
            seq_length,
            micro_batch_size,
            decoder_seq_length,
            config
    ):
        args = get_args()
        actual_micro_batch_size = getattr(args, "actual_micro_batch_size", None)
        micro_batch_size = micro_batch_size if actual_micro_batch_size is None else actual_micro_batch_size

        tensor_shape = get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config
         )

        return tensor_shape

    return wrapper


def load_checkpoint_loosely():
    args = get_args()
    return args.load_checkpoint_loosely