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

from typing import Union, List
import paddle
import torch
import numpy as np


class Compare():
    def __call__(self,
                 torch_model: torch.nn.Module,
                 paddle_model: paddle.nn.Layer,
                 torch_loss: torch.nn.Module,
                 paddle_loss: paddle.nn.Layer,
                 lr: float=1e-3,
                 steps: int=10,
                 input: Union[np.ndarray, dict]=None,
                 label: Union[np.ndarray, dict]=None,
                 input_shape: List[int]=None):
        # prepare torch
        torch_optim = torch.optim.SGD(params=torch_model.parameters(), lr=lr)
        optim = paddle.optimizer.SGD(parameters=paddle_model.parameters(),
                                     learning_rate=lr)
        torch_input, paddle_input = self.make_input(input, input_shape)

        for i in range(steps):
            outputs = torch_model(**inputs)

    def make_input(self,
                   input: Union[np.ndarray, dict]=None,
                   input_shape: List[int]=None):
        if input is None:
            assert input_shape is not None
            input = np.random.random(input_shape).astype(np.float32)
            torch_input = torch.Tensor(input)
            paddle_input = paddle.Tensor(input)
        else:
            if isinstance(input, dict):
                torch_input = {}
                paddle_input = {}
                for k, v in input.items():
                    if isinstance(v, np.ndarray):
                        torch_input[k] = torch.Tensor(v)
                        paddle_input[k] = paddle.Tensor(v)
                    else:
                        torch_input[k] = v
                        paddle_input[k] = v
            elif isinstance(input, np.ndarray):
                torch_input = torch.Tensor(input)
                paddle_input = paddle.Tensor(input)
            else:
                raise Exception('type of input must be numpy.ndarray or dict')
        return torch_input, paddle_input


if __name__ == "__main__":
    # paddle.save(model.state_dict(),'/Users/zhoujun20/Desktop/工作相关/VQA/PaddleNLP/contrib/layoutlm_re/model_state_from_torch_0,7077.pdparams')

    inputs = load_input(
        '/Users/zhoujun20/Desktop/工作相关/VQA/unilm/grad_input.pth')

    optim = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=5e-5,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0)

    for i in range(10):
        outputs = model(**inputs)
        outputs['loss'].backward()
        optim.step()
        optim.clear_grad()
        grad_dict = {}
        paddle_grad_dict = {}
        for name, parms in model.named_parameters():
            if not parms.stop_gradient and parms.grad is not None:
                paddle_grad_dict[name] = {
                    'grad_value': parms.grad,
                    'grad_requirs': parms.stop_gradient
                }
        torch_grad_dict = torch.load(
            f'/Users/zhoujun20/Desktop/工作相关/VQA/unilm/layoutlmft/grad/grad_{i}.pth'
        )
        match_keys = [
            "query.weight", "key.weight", "value.weight", "dense.weight",
            "rel_pos_bias", "rel_pos_x_bias", "rel_pos_y_bias", "visual_proj",
            'ffnn_head.0.weight', 'ffnn_head.3.weight', 'ffnn_tail.0.weight',
            'ffnn_tail.3.weight', 'rel_classifier.linear.weight'
        ]
        for key in torch_grad_dict:
            torch_grad = torch_grad_dict[key]['grad_value'].numpy()
            paddle_grad = paddle_grad_dict[key]['grad_value'].numpy()

            if any([s in key for s in match_keys]):
                torch_grad = torch_grad.transpose()

            diff = torch_grad - paddle_grad
            if diff.min() > 1e-6:
                print(i, key, diff.mean(), diff.max(), diff.min())

        print(i, outputs['loss'])
