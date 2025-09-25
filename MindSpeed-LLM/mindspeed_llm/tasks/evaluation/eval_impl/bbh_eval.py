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
import logging
import json
import re
import pandas as pd
import tqdm

from torch import distributed as dist
from .template import BBH_TEMPLATE_DIR
from ..eval_api.dataset_eval import DatasetEval
from ..eval_api.chat import Chat
from ...error_utils import check_divisible_by_zero

logger = logging.getLogger(__name__)


class BBHEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{fewshot_template}Q: {question}\nA:"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        total_acc_n = 0
        total_n = 0
        score_datas = []
        sample_n = 0
        rank = None

        with open(BBH_TEMPLATE_DIR, encoding='utf-8') as f:
            bbh_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if not os.path.exists(file_path):
                raise FileExistsError("The file ({}) does not exist !".format(file_path))
            
            with open(file_path, encoding='utf-8') as f:
                bbh_dataset = json.load(f)
            subject_name = re.sub(r'(?:_test|_val|_dev)?\.\w+$', "", file)
            subject_result = {}
            if 'examples' in bbh_dataset:
                sample_n += len(bbh_dataset['examples'])
            else:
                raise ValueError(f"key 'examples' not found in the bbh_dataset")
            sample_n += len(bbh_dataset['examples'])
            acc_n = 0
            sorted_dataset = sorted(bbh_dataset['examples'], key=lambda x: len(x['input']))
            instructions = []
            targets = []

            if subject_name not in bbh_template:
                logging.error(f"missing '{subject_name}' instruction_template in {BBH_TEMPLATE_DIR}")
                if self.file_pbar is not None:
                    self.file_pbar.update()
                continue

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(sorted_dataset), desc=file, leave=False)

            for idx, item in enumerate(sorted_dataset):
                instruction = self.instruction_template.format(fewshot_template=bbh_template[subject_name],
                                                               question=item['input'])
                instructions.append(instruction)
                targets.append(item['target'].lstrip('(').rstrip(')'))

                if len(instructions) == self.batch_size or len(bbh_dataset['examples']) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            try:
                                if rank == 0:
                                    logger.info("correct: %s, AI: %s", targets[index], answer.splitlines()[0])
                                    subject_result[str(idx - len(chat_results) + index + 1)] = answer.splitlines()[0]
                                    if subject_result[str(idx - len(chat_results) + index + 1)] == targets[index]:
                                        acc_n += 1
                            except Exception as e:
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + f". AI answer: {answer}"
                    instructions = []
                    targets = []

                if self.task_pbar is not None:
                    self.task_pbar.update()

            if rank == 0:
                logging.info(f"{subject_name} acc = {acc_n}/{len(bbh_dataset['examples'])}="
                             f"{check_divisible_by_zero(acc_n, len(bbh_dataset['examples']))}")
                total_n += len(bbh_dataset['examples'])
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(bbh_dataset['examples']),
                                    check_divisible_by_zero(acc_n, len(bbh_dataset['examples']))])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()        

        if rank == 0:
            logger.info(f"bbh acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, check_divisible_by_zero(total_acc_n, total_n)])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
