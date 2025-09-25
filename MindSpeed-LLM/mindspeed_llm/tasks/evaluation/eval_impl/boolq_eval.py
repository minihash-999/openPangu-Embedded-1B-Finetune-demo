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
import pandas as pd
import tqdm

from torch import distributed as dist
from ..eval_api.dataset_eval import DatasetEval
from ..eval_api.chat import Chat
from ...error_utils import check_divisible_by_zero


logger = logging.getLogger(__name__)


class BoolqEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{passage}\nQuestion: {question}?\nAnswer:"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        # BoolqEval not support n shot now.
        self.eval_template = None
        self.max_eval_samples = eval_args.max_eval_samples

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if not os.path.exists(file_path):
                raise FileExistsError("The file ({}) does not exist !".format(file_path))
            
            with open(file_path, encoding='utf-8') as f:
                boolq_question_list = []
                for line in f.readlines():
                    boolq_question_list.append(json.loads(line))
            subject_result = {}
            acc_n = 0
            instructions = []
            targets = []

            if self.max_eval_samples is not None:
                origin_len = len(boolq_question_list)
                boolq_question_list = (
                    boolq_question_list[0:min(self.max_eval_samples, origin_len)]
                )

                logger.info("%s length from %s to %s !!!", file, str(origin_len), str(len(boolq_question_list)))

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(boolq_question_list), desc=file, leave=False)

            for index, item in enumerate(boolq_question_list):
                instruction = self.instruction_template.format(passage=item['passage'], question=item['question'])
                instructions.append(instruction)
                targets.append(item['answer'])

                if len(instructions) == self.batch_size or len(boolq_question_list) == index + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if chat_results:
                        for idx, chat_result in enumerate(chat_results):
                            answer = chat_result[1].lstrip()
                            try:
                                if rank == 0:
                                    logger.info(f"correct: {str(targets[idx])[0]}, AI: {answer}")
                                    subject_result[str(index - len(chat_result) + idx + 1)] = answer
                                    if subject_result[str(index - len(chat_result) + idx + 1)] == str(targets[idx])[0]:
                                        acc_n += 1
                            except Exception as e:
                                if rank == 0:
                                    logger.info(e)
                                subject_result[str(index - len(chat_result) + idx + 1)] = str(
                                    e) + ". AI answer:" + answer
                    instructions = []
                    targets = []

                if self.task_pbar is not None:
                    self.task_pbar.update()

            if rank == 0:
                total_n += len(boolq_question_list)
                total_acc_n += acc_n
                answer_result['Boolq_dataset'] = subject_result
                score_datas.append(['Boolq_dataset', len(boolq_question_list), check_divisible_by_zero(acc_n, len(boolq_question_list))])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()

        if rank == 0:
            logger.info(f"boolq acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
