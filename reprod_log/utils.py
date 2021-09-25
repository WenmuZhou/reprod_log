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
import torch
import paddle
import numpy as np


def init_logger(save_path=None):
    """
    benchmark logger
    """
    # Init logger
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_output = save_path
    if not os.path.exists(os.path.dirname(log_output)):
        os.makedirs(os.path.dirname(log_output))
    if save_path is None:
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


def np2torch(data: dict):
    torch_input = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            torch_input[k] = torch.Tensor(v)
        else:
            torch_input[k] = v
    return torch_input


def np2paddle(data: dict):
    paddle_input = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            paddle_input[k] = paddle.Tensor(v)
        else:
            paddle_input[k] = v
    return paddle_input


def paddle2np(data):
    if isinstance(data, dict):
        np_data = {}
        for k, v in data.items():
            np_data[k] = v.numpy()
        return np_data
    else:
        return {'output': data.numpy()}


def torch2np(data):
    if isinstance(data, dict):
        np_data = {}
        for k, v in data.items():
            np_data[k] = v.detach().numpy()
        return np_data
    else:
        return {'output': data.detach().numpy()}


def print_diff(diff_dict, desc=''):
    print(f"{'*' * 10} {desc} {'*' * 10}")
    for k, v in diff_dict.items():
        print(k, f"diff max: {v['diff'].max()},diff mean: {v['diff'].mean()}")
