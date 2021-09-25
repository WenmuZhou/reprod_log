# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import torch
import paddle
import numpy as np

from .utils import np2torch, np2paddle, paddle2np, torch2np


def check_data(data1: dict, data2: dict):
    for k in data1:
        if k not in data2:
            assert k in data2, 'k in data1 but not found in data2'.format(
                k, data2)

    for k in data2:
        if k not in data1:
            assert k in data1, 'k in data2 but not found in data1'.format(
                k, data2.keys())


def compute_diff(data1: dict, data2: dict, indent='\t'):
    out_dict = {}
    for k in data1:
        assert k in data2
        sub_data1, sub_data2 = data1[k], data2[k]
        assert type(sub_data1) == type(sub_data2)
        if isinstance(sub_data1, dict):
            out = compute_diff(sub_data1, sub_data2, indent)
            for sub_k, sub_v in out.items():
                out_dict[f'{k}{indent}{sub_k}'] = sub_v
        elif isinstance(sub_data1, np.ndarray):
            diff = np.abs(sub_data1 - sub_data2)
            out_dict[k] = {'out1': sub_data1, 'out2': sub_data2, 'diff': diff}
        else:
            raise NotImplementedError
    return out_dict


def compare_model(torch_model: torch.nn.Module,
                  paddle_model: paddle.nn.Layer,
                  input: dict=None):
    torch_input = np2torch(input)
    paddle_input = np2paddle(input)

    torch_model.eval()
    paddle_model.eval()
    torch_out = torch_model(**torch_input)
    paddle_out = paddle_model(**paddle_input)

    diff = compute_diff(torch2np(torch_out), paddle2np(paddle_out))
    return diff
