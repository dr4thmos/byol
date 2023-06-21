# mean
# std
# logsoftmax?

import torch

m = torch.nn.LogSoftmax(dim=1)
input = torch.randn(2, 3)
output = m(input)