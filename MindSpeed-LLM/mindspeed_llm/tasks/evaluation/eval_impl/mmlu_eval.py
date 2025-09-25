# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import logging
import json
import pandas as pd
import tqdm

from torch import distributed as dist
from .template import MMLU_TEMPLATE_DIR, get_eval_template
from ..eval_api.dataset_eval import DatasetEval
from ..eval_api.chat import Chat
from ...error_utils import check_divisible_by_zero

logger = logging.getLogger(__name__)


class MmluEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{few_shot_examples}\n\n"
                                      "{question}\nAnswer:",
                 output_template1=r".*(?P<answer>[A|B|C|D])\..*",
                 output_template2=r"(?P<answer>[A|B|C|D])"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.output_template = [output_template1, output_template2]
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.eval_template = None
        self.prompt_type = None
        if eval_args.prompt_type is not None:
            self.prompt_type = eval_args.prompt_type.strip()
            self.eval_template = get_eval_template(eval_args.eval_language)
        self.max_eval_samples = eval_args.max_eval_samples

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None
        with open(MMLU_TEMPLATE_DIR, encoding='utf-8') as f:
            mmlu_few_shot_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if os.path.exists(file_path):
                data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
            else:
                raise FileNotFoundError(f"Error: {file_path} does not exist.")
            
            subject_name = re.sub(r'(?:_test|_val|_dev)?\.\w+$', "", file)  # 文件命名规则类似  {subject}_test.csv
            subject = subject_name.replace("_", " ")
            subject_result = {}
            acc_n = 0
            instructions = []
            corrects = []

            if self.max_eval_samples is not None:
                origin_len = len(data_df)
                data_df = (
                    data_df[0:min(self.max_eval_samples, origin_len)]
                )

                logger.info("%s length from %s to %s !!!", subject_name, str(origin_len), str(len(data_df)))

            if subject_name not in mmlu_few_shot_template:
                logging.error(f"missing '{subject_name}' instruction_template in {MMLU_TEMPLATE_DIR}")
                if self.file_pbar is not None:
                    self.file_pbar.update()
                continue

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(data_df), desc=file, leave=False)

            for idx, row in data_df.iterrows():
                instruction = None
                if self.prompt_type is not None:
                    normalized_test_dir = os.path.normpath(self.test_dir)
                    train_dir = os.path.join(os.path.dirname(normalized_test_dir), "dev")
                    train_file_path = os.path.join(train_dir, subject_name + "_dev.csv")

                    if not os.path.exists(train_file_path):
                        raise FileExistsError("The file ({}) does not exist !".format(train_file_path))

                    train_data_df = pd.read_csv(train_file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
                    support_set = (
                        train_data_df.sample(min(5, len(train_data_df)))
                    )
                    instruction = self.eval_template.format_example(
                        target_data=row,
                        support_set=support_set,
                        subject_name=subject_name,
                    )
                else:
                    test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                    instruction = self.instruction_template.format(few_shot_examples=mmlu_few_shot_template[subject_name],
                                                                subject=subject,
                                                                question=test_question)
                instructions.append(instruction)
                corrects.append(row['answer'])

                if len(instructions) == self.batch_size or len(data_df) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            try:
                                if rank == 0:
                                    logger.info(instruction)
                                    match_flag = False
                                    for template_idx, template in enumerate(self.output_template):
                                        try:
                                            result = re.match(template, answer)
                                            logger.info(f"correct: {corrects[index]}, AI: {result.group('answer')}")
                                            subject_result[str(idx - len(chat_results) + index + 1)] = result.group(
                                                "answer")
                                            if subject_result[str(idx - len(chat_results) + index + 1)] == corrects[
                                                index]:
                                                acc_n += 1
                                            match_flag = True
                                            break
                                        except Exception as e:
                                            if template_idx == len(self.output_template) - 1:
                                                logger.info(e)
                                            continue
                                    if not match_flag:
                                        logger.info("xx. AI answer: %s", answer)
                            except Exception as e:
                                if rank == 0:
                                    logger.info(e)
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + ". AI answer:" + answer
                    instructions = []
                    corrects = []

                if self.task_pbar is not None:
                    self.task_pbar.update()

            if rank == 0:
                total_n += len(data_df)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(data_df), acc_n / len(data_df)])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()        

        if rank == 0:
            logger.info(f"mmlu acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
