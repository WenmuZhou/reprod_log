# reprod_log

主要用于对比和记录模型复现过程中的各个步骤精度对齐情况

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
