import torch
from torch import tensor

a = tensor([[1,2,3,4],[5,6,7,8]])
print(torch.reshape(a, (4, 2)))