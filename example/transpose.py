import torch

a = [[1,2,3],[4,5,6]]
a = torch.as_tensor(a, dtype=float)
print(a, a.dtype)

# help(torch.transpose)

b = a.transpose(0, 1)
print(b, b.dtype)