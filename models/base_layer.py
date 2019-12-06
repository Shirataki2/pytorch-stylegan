import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnect(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        gain=2**.5,
        use_wscale=False,
        lrmul=1.,
        bias=True
    ):
        super().__init__()
        # He Initalizer
        he_std = gain * in_channels ** -.5
        if use_wscale:
            init_std = 1 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels
            ) * init_std
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            o = F.linear(x, self.weight * self.w_mul,
                         self.bias * self.b_mul)
        else:
            o = F.linear(x, self.weight * self.w_mul)
        o = F.leaky_relu(o, 0.2, inplace=True)
        return o


class Blur2d(nn.Module):
    def __init__(self, f=[1, 2, 1], normalize=True, flip=False, stride=1):
        super().__init__()
        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                f = torch.flip(f, [2, 3])
        self.f = f
        self.stride = stride

    def forward(self, x):
        if self.f is None:
            return x
        k = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
        x = F.conv2d(
            x,
            k,
            stride=self.stride,
            padding=int((self.f.size(2)-1)/2),
            groups=x.size(1)
        )
        return x


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        gain=2**.5,
        use_wscale=False,
        lrmul=1.,
        bias=True
    ):
        super().__init__()
        # He Initalizer
        he_std = gain * (kernel_size**2 * in_channels) ** -.5
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            ) * init_std
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            o = F.conv2d(x, self.weight * self.w_mul,
                         self.bias * self.b_mul, padding=self.kernel_size//2)
        else:
            o = F.conv2d(x, self.weight * self.w_mul,
                         padding=self.kernel_size//2)
        return o


class PixelNormalization(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        sq = torch.mul(x, x)
        msr = torch.rsqrt(torch.mean(sq, dim=1, keepdim=True) + self.eps)
        return x * msr


class InstanceNormalization(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        var = torch.mul(x, x)
        std = torch.rsqrt(torch.mean(var, (2, 3), True) + self.eps)
        return x * std


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1.):
        super().__init__()
        self.g = gain
        self.f = factor

    def forward(self, x):
        if self.g != 1:
            x = x * self.g
        if self.f > 1:
            s = x.shape
            x = x.view(
                s[0], s[1], s[2], 1, s[3], 1
            ).expand(-1, -1, -1, self.f, -1, self.f)
            x = x.contiguous().view(s[0], s[1], self.f * s[2], self.f * s[3])
        return x
