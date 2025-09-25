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
from .template import AGIEVAL_TEMPLATE_DIR
from ..eval_api.dataset_eval import DatasetEval
from ..eval_api.chat import Chat
from ...error_utils import check_divisible_by_zero

logger = logging.getLogger(__name__)


class AGIEvalExam(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{fewshot_template}   {question}\n{question_template}\n{options}"
                                      "\n{answer_template}"):
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
        
        with open(AGIEVAL_TEMPLATE_DIR, encoding='utf-8') as f:
            AGI_few_shot_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            with open(file_path, encoding='utf-8') as f:
                agi_question_list = []
                for line in f.readlines():
                    agi_question_list.append(json.loads(line))
            subject_name = re.sub(r'(?:_test|_val|_dev)?\.\w+$', "", file)
            subject_result = {}
            sample_n += len(agi_question_list)
            acc_n = 0
            instructions = []
            corrects = []

            if subject_name not in AGI_few_shot_template:
                logging.error(f"missing '{subject_name}' instruction_template in {AGIEVAL_TEMPLATE_DIR}")
                if self.file_pbar is not None:
                    self.file_pbar.update()
                continue

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(agi_question_list), desc=file, leave=False)

            for idx, item in enumerate(agi_question_list):
                if item['passage']:
                    question = item['passage'] + '\n' + item['question']
                else:
                    question = item['question']
                if item['options']:
                    options = '\n'.join(item['options'])
                else:
                    options = ""
                if item['label']:
                    if isinstance(item['label'], list):
                        correct = ','.join(item['label'])
                    else:
                        correct = item['label']
                else:
                    if item['answer']:
                        correct = item['answer'].replace('$', '')
                    else:
                        correct = None
                instruction = self.instruction_template.format(fewshot_template=AGI_few_shot_template[subject_name][0],
                                                               question=question,
                                                               question_template=AGI_few_shot_template[subject_name][1],
                                                               options=options,
                                                               answer_template=AGI_few_shot_template[subject_name][2])
                instructions.append(instruction)
                corrects.append(correct)

                if len(instructions) == self.batch_size or len(agi_question_list) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            try:
                                if rank == 0:
                                    final_result = answer.splitlines()[0].replace('$', '').replace('(', '').replace(')',
                                                                                                                    '')
                                    logger.info("correct: %s, AI: %s", corrects[index], final_result)
                                    subject_result[str(idx - len(chat_results) + index + 1)] = final_result
                                    if subject_result[str(idx - len(chat_results) + index + 1)] == corrects[index]:
                                        acc_n += 1
                            except Exception as e:
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + f". AI answer: {answer}"
                    instructions = []
                    corrects = []

                if self.task_pbar is not None:
                    self.task_pbar.update()

            if rank == 0:
                logger.info("%s acc = %d/%d=%e", subject_name, acc_n, len(agi_question_list),
                            check_divisible_by_zero(acc_n, len(agi_question_list)))
                total_n += len(agi_question_list)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append(
                    [subject_name, len(agi_question_list), check_divisible_by_zero(acc_n, len(agi_question_list))])
        
            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()        
        
        if rank == 0:
            logger.info("AGIEval acc = %d/%d=%e", total_acc_n, total_n, check_divisible_by_zero(total_acc_n, total_n))
            score_datas.append(["total", total_n, check_divisible_by_zero(total_acc_n, total_n)])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
