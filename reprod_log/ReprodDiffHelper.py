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
import sys

import numpy as np

from utils import init_logger
from compare import compute_diff


class ReprodDiffHelper:
    def load_info(self, path):
        """
        加载字典文件
        :param path:
        :return:
        """
        data = np.load(path, allow_pickle=True).tolist()
        return data

    def compare_info(self, info1, info2):
        """
        对比diff
        :param info1:
        :param info2:
        :return:
        """
        self.diff = compute_diff(info1, info2)

    def report(self, diff_threshold=1e-6, path=None):
        """
        可视化diff，保存到文件或者到屏幕
        :param diff_threshold:
        :param path:
        :return:
        """
        logger = init_logger(path)

        logger.info("{}".format('*' * 20))
        passed = True
        for k, v in self.diff.items():
            mean_value = v['diff'].mean()
            logger.info("{}\t, diff mean: {}".format(k, mean_value))
            if mean_value > diff_threshold:
                logger.error('diff in {} failed the acceptance'.format(k))
                passed = False
                sys.exit(0)
        if passed:
            logger.info('passed')
