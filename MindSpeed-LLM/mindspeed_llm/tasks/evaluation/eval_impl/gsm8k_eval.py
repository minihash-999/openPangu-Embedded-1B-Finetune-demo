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
from .template import GSM8K_TEMPLATE_DIR
from ..eval_api.dataset_eval import DatasetEval
from ..eval_api.chat import Chat
from ...error_utils import check_divisible_by_zero

logger = logging.getLogger(__name__)


class Gsm8kEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{fewshot_template}\n\n{question}",
                 output_template=r'The answer is (.*?) '):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.output_template = output_template
        self.batch_size = eval_args.evaluation_batch_size      
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        final_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None
        
        with open(GSM8K_TEMPLATE_DIR, encoding='utf-8') as f:
            gsm8k_few_shot_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: {file_path} does not exist.")
            
            with open(file_path, encoding='utf-8') as f:
                gsm8k_list = []
                for line in f.readlines():
                    gsm8k_list.append(json.loads(line))
            subject_result = {}
            acc_n = 0
            instructions = []
            answers = []

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(gsm8k_list), desc=file, leave=False)

            for index, item in enumerate(gsm8k_list):
                instruction = self.instruction_template.format(fewshot_template=gsm8k_few_shot_template['few_shot'],
                                                               question=item['question'])
                instructions.append(instruction)
                answers.append([item['answer'].split('#### ')[-1]])
                if len(instructions) == self.batch_size or len(gsm8k_list) == index + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])

                    if chat_results:
                        for idx, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            answer = answer.split('Q:')[0]
                            answer_result = answer.replace('$', '').replace(',', '') + '  '
                            answer_result = answer_result.replace('.', ' ', -1)

                            try:
                                if rank == 0:
                                    logger.info(instruction)
                                    final_answer = re.findall(self.output_template, answer_result)
                                    final_answer = [final_answer[0][::-1].replace('.', '', 1)[::-1]]
                                    logger.info("correct: %s, AI: %s", answers[idx], final_answer)
                                    subject_result[str(index - len(chat_results) + idx + 1)] = final_answer
                                    if subject_result[str(index - len(chat_results) + idx + 1)] == answers[idx]:
                                        acc_n += 1
                            except Exception as e:
                                if rank == 0:
                                    logger.info(e)
                                subject_result[str(index - len(chat_results) + idx + 1)] = str(
                                    e) + ". AI answer:" + answer

                    instructions = []
                    answers = []

                if self.task_pbar is not None:
                    self.task_pbar.update()

            if rank == 0:
                total_n += len(gsm8k_list)
                total_acc_n += acc_n
                final_result['Gsm8k_dataset'] = subject_result
                score_datas.append(['Gsm8k_dataset', len(gsm8k_list), acc_n / len(gsm8k_list)])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()        

        if rank == 0:
            logger.info(f"gsm8k acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return final_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
