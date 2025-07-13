import torch
from torch import tensor

a = tensor([[1, 2], [2, 4]], dtype=torch.float32)
b = tensor([1, 2], dtype=torch.float32)

print(a, a.dtype)
print(b, b.dtype)

c = torch.matmul(a, b)
print(c, c.dtype)