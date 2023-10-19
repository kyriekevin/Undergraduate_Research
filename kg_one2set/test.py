# -*- coding: utf-8 -*-
# @Author: zyz
# @Time: 2021/8/22 10:14
# @File: test.py
# @Software: PyCharm

import torch

x = torch.rand((2, 2))
print(x)
for i in range(x.size(0)):
    print(x[i, i])
print(x.sum(axis=0))