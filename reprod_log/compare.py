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
import sys

import torch
import paddle
import numpy as np

from .utils import np2torch, np2paddle, paddle2np, torch2np, print_diff


def check_data(data1: dict, data2: dict):
    for k in data1:
        if k not in data2:
            assert k in data2, 'k in data1 but not found in data2'.format(
                k, data2)

    for k in data2:
        if k not in data1:
            assert k in data1, 'k in data2 but not found in data1'.format(
                k, data2.keys())


def compute_diff(data1: dict, data2: dict, indent: str='\t'):
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
            if sub_data1.shape != sub_data2.shape and sub_data1.transpose(
            ).shape == sub_data2.shape:
                print('transpose sub_data1')
                sub_data1 = sub_data1.transpose()
            diff = np.abs(sub_data1 - sub_data2)
            out_dict[k] = {'out1': sub_data1, 'out2': sub_data2, 'diff': diff}
        else:
            raise NotImplementedError
    return out_dict


def compare_forward(torch_model: torch.nn.Module,
                    paddle_model: paddle.nn.Layer,
                    input_dict: dict,
                    diff_threshold: float=1e-6):
    torch_input = np2torch(input_dict)
    paddle_input = np2paddle(input_dict)

    torch_model.eval()
    paddle_model.eval()
    torch_out = torch_model(**torch_input)
    paddle_out = paddle_model(**paddle_input)

    diff = compute_diff(torch2np(torch_out), paddle2np(paddle_out))
    passed = print_diff(diff, diff_threshold)
    if passed:
        print('Check passed')
    else:
        print('Check not passed')


def compare_loss_and_backward(torch_model: torch.nn.Module,
                              paddle_model: paddle.nn.Layer,
                              torch_loss: torch.nn.Module,
                              paddle_loss: paddle.nn.Layer,
                              input_dict: dict,
                              lr: float=1e-3,
                              steps: int=10,
                              diff_threshold: float=1e-6):
    torch_input = np2torch(input_dict)
    paddle_input = np2paddle(input_dict)

    torch_model.eval()
    paddle_model.eval()

    torch_optim = torch.optim.SGD(params=torch_model.parameters(), lr=lr)
    paddle_optim = paddle.optimizer.SGD(parameters=paddle_model.parameters(),
                                        learning_rate=lr)

    for i in range(steps):
        # paddle
        paddle_outputs = paddle_model(**paddle_input)
        paddle_loss_value = paddle_loss(paddle_input, paddle_outputs)
        paddle_loss_value['loss'].backward()
        paddle_optim.step()
        paddle_optim.clear_grad()

        paddle_grad_dict = {'loss': paddle_loss_value['loss'].numpy()}
        for name, parms in paddle_model.named_parameters():
            if not parms.stop_gradient and parms.grad is not None:
                paddle_grad_dict[name] = parms.grad.numpy()

        # torch
        torch_outputs = torch_model(**torch_input)
        torch_loss_value = torch_loss(torch_input, torch_outputs)
        torch_loss_value['loss'].backward()
        torch_optim.step()
        torch_optim.zero_grad()

        torch_grad_dict = {'loss': torch_loss_value['loss'].detach().numpy()}
        for name, parms in torch_model.named_parameters():
            if parms.requires_grad and parms.grad is not None:
                torch_grad_dict[name] = parms.grad.numpy()

        # compare
        diff = compute_diff(paddle_grad_dict, torch_grad_dict)
        passed = print_diff(diff, diff_threshold)
        if not passed:
            print('Check not passed as iter {}'.format(i))
            sys.exit()
    print('Check passed')
