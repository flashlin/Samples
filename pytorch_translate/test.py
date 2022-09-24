import torch

a = torch.Tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

b = a[1:]
print(f"{b=}")
