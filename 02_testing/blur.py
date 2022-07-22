from __future__ import annotations
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from torch import Tensor

def separable_conv2d(inputs: Tensor, k_h: Tensor, k_w: Tensor) -> Tensor:
    kernel_size = max(k_h.shape[-2:])
    
    pad_amount = kernel_size // 2 #'same' padding.
    # Gaussian filter is separable:
    out_1 = F.conv2d(inputs, k_h, padding=(0, pad_amount))
    out_2 = F.conv2d(out_1, k_w, padding=(pad_amount, 0))
    return out_2

class GaussianBlur2d(nn.Module):
    def __init__(self, std: float = 2.0, trainable_std=False):
        super().__init__()
        self.std = Parameter(torch.as_tensor(std), requires_grad=trainable_std)

    def forward(self, inputs: Tensor) -> Tensor:
        #smallest image dimension 
        image_size = min(inputs.shape[-2:])
        
        k = kernel(self.std, image_size)
        k_h = k.view([1,1,1,-1]).type_as(inputs)
        k_w = k_h.transpose(-1, -2)

        kernel_size = k_h.shape[-1]
        pad_amount = kernel_size // 2 #'same' padding.
        # Gaussian filter is separable:
        out_1 = F.conv2d(inputs, k_h, padding=(0, pad_amount))
        out_2 = F.conv2d(out_1, k_w, padding=(pad_amount, 0))
        return out_2


def gaussian_kernel_1d(
    std: float | Tensor,
    kernel_size: int,
    dtype: torch.dtype = torch.float32) -> Tensor:
    exponents = [range(- kernel_size // 2 + 1, kernel_size // 2 + 1)]
    x = torch.as_tensor(exponents, dtype=dtype)
    g = torch.exp(- (x**2 / (2 * std**2)) / (np.sqrt(2 * np.pi) * std))
    # normalize the sum to 1
    g = g / g.sum()
    return g


def kernel(std: Tensor, image_size: int) -> Tensor:
    """ Creates the kernel dynamically depending on the std. """
    k = kernel_size(std=float(std), image_size=image_size)
    return gaussian_kernel_1d(std=std, kernel_size=k)


def kernel_size(std: float, image_size: int | None = None) -> int:
    """
    Determines the kernel size dynamically depending on the std.
    We limit the kernel size to the smallest image dimension, at most.
    """
    # nearest odd number to 5*std.
    k_size = odd_integer_above(5 * std)
    # the kernel shouldn't be smaller than 3.
    k_size = max(k_size, 3)
    
    if image_size:
        # can't have kernel bigger than image size.
        max_k_size = odd_integer_below(image_size)
        k_size = min(k_size, max_k_size)

    assert k_size % 2 == 1, "kernel size should be odd"
    return k_size

def odd_integer_above(number: float) -> int:
    integer = int(np.ceil(number))
    return integer if integer % 2 == 1 else integer+1

def odd_integer_below(number: float) -> int:
    return odd_integer_above(number) - 2


def main():
    from torchvision import datasets
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from typing import Literal as L
    bob: Tensor[L, L[32]]

    data_dir = "~/scratch/data"
    batch_size = 32
    mnist_transforms = [transforms.ToTensor()]
    use_cuda = torch.cuda.is_available()
    print("use cuda:", use_cuda)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_dir,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        ), 
        batch_size=32, **kwargs
    )

    blur = GaussianBlur()

    for step, (image_batch, label_batch) in enumerate(dataset):
        plt.imshow(image_batch[0, 0])
        image_batch.numpy()
        blurred = blur(image_batch)
        print(blurred.shape)
        plt.imshow(blurred[0, 0])
        plt.show()
        break