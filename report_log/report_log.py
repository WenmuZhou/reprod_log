# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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


class ReportLog(object):
    def __init__(self, save_path=None, logger=None):
        self.save_path = save_path
        self.logger = self.init_logger() if logger is None else logger

    def init_logger(self):
        """
        benchmark logger
        """
        # Init logger
        FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_output = f"{self.save_path}"
        if not os.path.exists(os.path.dirname(log_output)):
            os.makedirs(os.path.dirname(log_output))
        if self.save_path is None:
            logging.basicConfig(
                level=logging.INFO,
                format=FORMAT)
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=FORMAT,
                handlers=[
                    logging.FileHandler(
                        filename=log_output, mode='w'),
                    logging.StreamHandler(),
                ])
        logger = logging.getLogger(__name__)
        logger.info("Init logger done!")
        return logger

    def compare(self, data1: dict, data2: dict, key: str = None):
        """
        对比两个字典的数据diff
        :param data1:
        :param data2:
        :param key: 当前对比数据的描述
        :return:
        """
        pass

    def compare_model(self, paddle_model, torch_model, input=None, input_shape=None):
        pass

    def compare_loss(self, paddle_model, torch_model, input=None, input_shape=None):
        pass

    def compare_grad(self, paddle_model, torch_model, input=None, input_shape=None):
        pass