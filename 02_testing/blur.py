from __future__ import annotations
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from torch import Tensor


def gaussian_kernel_1d(
    std: float | Tensor, size: int, dtype: torch.dtype = torch.float32
) -> Tensor:
    exponents = [range(-size // 2 + 1, size // 2 + 1)]
    x = torch.as_tensor(exponents, dtype=dtype)
    g = torch.exp(-(x**2 / (2 * std**2)) / (np.sqrt(2 * np.pi) * std))
    # normalize the sum to 1
    g = g / g.sum()
    return g


def separable_conv2d(inputs: Tensor, k_h: Tensor, k_w: Tensor) -> Tensor:
    k = max(k_h.shape[-2:])
    assert k % 2 == 1
    assert k_h.shape[-2:] == (1, k)
    assert k_w.shape[-2:] == (k, 1)
    pad_amount = k // 2  #'same' padding.
    out = F.conv2d(inputs, k_h, padding="same")
    out = F.conv2d(out, k_w, padding="same")
    return out


def separable_conv3d(x: Tensor, h_k: Tensor, w_k: Tensor, d_k: Tensor) -> Tensor:
    k = max(h_k.shape[-3:])
    assert k % 2 == 1, "kernel size should be odd"
    assert h_k.shape[-3:] == (1, 1, k), h_k.shape
    assert w_k.shape[-3:] == (1, k, 1), w_k.shape
    assert d_k.shape[-3:] == (k, 1, 1), d_k.shape
    o = F.conv3d(x, weight=h_k, padding="same")
    o = F.conv3d(o, weight=w_k, padding="same")
    o = F.conv3d(o, weight=d_k, padding="same")
    return o


class GaussianBlur2d(nn.Module):
    def __init__(self, std: float = 2.0, trainable: bool = False):
        super().__init__()
        self.std = Parameter(torch.as_tensor(std), requires_grad=trainable)

    def forward(self, inputs: Tensor) -> Tensor:
        image_size = min(inputs.shape[-2:])
        k_size = kernel_size(std=float(self.std), image_size=image_size)
        k = gaussian_kernel_1d(std=self.std, size=k_size)
        k_h = k.view([1, 1, 1, -1]).type_as(inputs)
        k_w = k_h.transpose(-1, -2)
        return separable_conv2d(inputs, k_h, k_w)


class GaussianBlur3d(nn.Module):
    def __init__(self, std: float = 2.0, trainable: bool = False):
        super().__init__()
        self.std = Parameter(torch.as_tensor(std), requires_grad=trainable)

    def forward(self, inputs: Tensor) -> Tensor:
        image_size = min(inputs.shape[-2:])
        k_size = kernel_size(std=float(self.std), image_size=image_size)
        k = gaussian_kernel_1d(std=self.std, size=k_size)
        k_h = k.view([1, 1, 1, -1]).type_as(inputs)
        k_w = k_h.transpose(-1, -2)
        k_d = k_h.transpose(-1, -3)
        return separable_conv3d(inputs, k_h, k_w, k_d)


class SeparableGaussianBlur2d(nn.Module):
    """Variant where a different std is used for the height and width of the kernel."""

    def __init__(
        self,
        std: float = 2.0,
        std_h: float | None = None,
        std_w: float | None = None,
        trainable=False,
    ):
        super().__init__()
        std_h = std if std_h is None else std_h
        std_w = std if std_w is None else std_w
        self.std_h = Parameter(torch.as_tensor(std_h), requires_grad=trainable)
        self.std_w = Parameter(torch.as_tensor(std_w), requires_grad=trainable)

    def forward(self, inputs: Tensor) -> Tensor:
        image_size = min(inputs.shape[-2:])
        h_k_size = kernel_size(std=float(self.std_h), image_size=image_size)
        w_k_size = kernel_size(std=float(self.std_w), image_size=image_size)
        h_k = gaussian_kernel_1d(std=self.std_h, size=h_k_size)
        w_k = gaussian_kernel_1d(std=self.std_w, size=w_k_size)
        h_k = h_k.view([1, 1, 1, -1]).type_as(inputs)
        w_k = w_k.view([1, 1, -1, 1]).type_as(inputs)
        return separable_conv2d(inputs, h_k, w_k)


class SeparableGaussianBlur3d(nn.Module):
    """Variant where a different std is used for the height, width and depth of the kernel."""

    def __init__(
        self,
        std: float = 2.0,
        std_h: float | None = None,
        std_w: float | None = None,
        std_d: float | None = None,
        trainable=False,
    ):
        super().__init__()
        std_h = std if std_h is None else std_h
        std_w = std if std_w is None else std_w
        std_d = std if std_d is None else std_d
        self.std_h = Parameter(torch.as_tensor(std_h), requires_grad=trainable)
        self.std_w = Parameter(torch.as_tensor(std_w), requires_grad=trainable)
        self.std_d = Parameter(torch.as_tensor(std_d), requires_grad=trainable)

    def forward(self, inputs: Tensor) -> Tensor:
        image_size = min(inputs.shape[-2:])
        h_k_size = kernel_size(std=float(self.std_h), image_size=image_size)
        w_k_size = kernel_size(std=float(self.std_w), image_size=image_size)
        d_k_size = kernel_size(std=float(self.std_d), image_size=image_size)
        h_k = gaussian_kernel_1d(std=self.std_h, size=h_k_size)
        w_k = gaussian_kernel_1d(std=self.std_w, size=w_k_size)
        d_k = gaussian_kernel_1d(std=self.std_d, size=d_k_size)
        h_k = h_k.view([1, 1, 1, -1]).type_as(inputs)
        w_k = w_k.view([1, 1, -1, 1]).type_as(inputs)
        d_k = w_k.view([1, -1, 1, 1]).type_as(inputs)
        return separable_conv3d(inputs, h_k, w_k, d_k)


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

    assert k_size % 2 == 1
    return k_size


def odd_integer_above(number: float) -> int:
    integer = int(np.ceil(number))
    return integer if integer % 2 == 1 else integer + 1


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
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    dataset = DataLoader(
        datasets.MNIST(
            data_dir,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=32,
        **kwargs,
    )

    blur = GaussianBlur2d()

    for step, (image_batch, label_batch) in enumerate(dataset):
        plt.imshow(image_batch[0, 0])
        image_batch.numpy()
        blurred = blur(image_batch)
        print(blurred.shape)
        plt.imshow(blurred[0, 0])
        plt.show()
        break


if __name__ == "__main__":
    main()
