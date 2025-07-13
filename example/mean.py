import torch
from torch import tensor

a = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64)
b = torch.mean(a, dim = 0)
print(b)