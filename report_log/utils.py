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
import torch
import paddle
import numpy as np


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


def compare_model(torch_model: torch.nn.Module,
                  paddle_model: paddle.nn.Layer,
                  input: dict = None):
    torch_input = np2torch(input)
    paddle_input = np2paddle(input)
    torch_out = torch_model(**torch_input)
    paddle_out = paddle_model(**paddle_input)

    diff = compute_diff(torch2np(torch_out), paddle2np(paddle_out))
    return diff
