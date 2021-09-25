# reprod_log

主要用于对比和记录模型复现过程中的各个步骤精度对齐情况
## 安装

1. 本地编译安装
```bash
python3 setup.py bdist_wheel
python3 install dist/reprod_log-x.x.-py3-none-any.whl --force-reinstall
```

2. pip直接安装
```bash
# from pypi
pip3 install reprod_log --force-reinstall
# from github, 取决于网络环境
pip3 install git+https://github.com/WenmuZhou/reprod_log.git --force-reinstall
```
## 提供的类和方法

类 `ReprodLogger` 用于记录和报错复现过程中的中间变量

主要方法为

* add(key, val)：添加key-val pair
* remove(key)：移除key
* clear()：清空字典
* save(path)：保存字典

类 `ReprodDiffHelper` 用于对中间变量进行检查，主要为计算diff

主要方法为

* load_info(path): 加载字典文件
* compare_info(info1:dict, info2:dict): 对比diff
* report(diff_threshold=1e-6,path=None): 可视化diff，保存到文件或者到屏幕

`compare` 模块提供了基础的网络前向和反向过程对比工具

* compare_forward 用于对比网络的反向过程，其参数为
  * torch_model: torch.nn.Module,
  * paddle_model: paddle.nn.Layer,
  * input_dict: dict, dict值为numpy矩阵
  * diff_threshold: float=1e-6

* compare_loss_and_backward 用于对比网络的反向过程，其参数为
  * torch_model: torch.nn.Module,
  * paddle_model: paddle.nn.Layer,
  * torch_loss: torch.nn.Module,
  * paddle_loss: paddle.nn.Layer,
  * input_dict: dict, dict值为numpy矩阵
  * lr: float=1e-3,
  * steps: int=10,
  * diff_threshold: float=1e-6
