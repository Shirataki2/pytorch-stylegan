import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_layer import FullyConnect, Conv2d, DepthwiseConv2d, InstanceNormalization, Upscale2d, PixelNormalization
from models.gans_util import ApplyNoise, ApplyStyle


class LayerMixtureEpilogue(nn.Module):
    def __init__(
        self,
        channels,
        dlatent_size,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instanse_norm,
        use_styles
    ):
        super().__init__()
        self.layers = []
        if use_noise:
            self.apply_noise = ApplyNoise(channels)
        else:
            self.apply_noise = None
        self.layers.append(nn.LeakyReLU(negative_slope=.2))
        if use_pixel_norm:
            self.layers.append(PixelNormalization())
        if use_instanse_norm:
            self.layers.append(InstanceNormalization())
        self.layers = nn.ModuleList(self.layers)
        if use_styles:
            self.apply_style = ApplyStyle(dlatent_size, channels, use_wscale)
        else:
            self.apply_style = None

    def forward(self, x, noise, dlatent_slice=None):
        if self.apply_noise:
            x = self.apply_noise(x)
        for layer in self.layers:
            x = layer(x)
        if self.apply_style:
            x = self.apply_style(x, dlatent_slice)
        return x


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        resolution,  # 2 ** resolution
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instanse_norm,
        use_specnorm,
        noise_input,
        dlatent_size=512,
        use_styles=True,
        lowpass_filter=None,
        scale_factor=2,
        fmap=lambda x: min(int(2**13 / (2.**x)), 512)
    ):
        super().__init__()
        self.res = resolution
        self.blur = DepthwiseConv2d(lowpass_filter)
        self.noise_input = noise_input
        if self.res > 6:
            self.upsample = nn.ConvTranspose2d(
                fmap(self.res-3), fmap(self.res-2), 4, stride=2, padding=1
            )
        else:
            self.upsample = Upscale2d(scale_factor)
        self.adaIn1 = LayerMixtureEpilogue(
            fmap(self.res-2), dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instanse_norm, use_styles
        )
        self.conv1 = Conv2d(
            fmap(self.res-2), fmap(self.res-2), 3, use_wscale=use_wscale)
        if use_specnorm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.adaIn2 = LayerMixtureEpilogue(
            fmap(self.res-2), dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instanse_norm, use_styles
        )

    def forward(self, x, w):
        n = self.noise_input
        r = self.res
        x = self.upsample(x)
        x = self.adaIn1(x, n[r*2-4], w[:, r*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, n[r*2-3], w[:, r*2-3])
        return x


class GeneratorMapper(nn.Module):
    def __init__(
        self,
        resolution=1024,
        z_latent_dim=512,
        w_latent_dim=512,
        normalize=True,
        use_wscale=True,
        lrmul=.01,
        gain=2**.5
    ):
        super().__init__()
        self.zdim = z_latent_dim
        self.fc = nn.Sequential(
            FullyConnect(self.zdim, w_latent_dim, gain,
                         use_wscale, lrmul),
            *([FullyConnect(w_latent_dim, w_latent_dim, gain, use_wscale, lrmul)] * 7)
        )
        self.normalize = normalize
        self.num_layers = int(np.log2(resolution)) * 2 - 2
        self.pixel_norm = PixelNormalization()

    def forward(self, z):
        if self.normalize:
            z = self.pixel_norm(z)
        w = self.fc(z)
        return w, self.num_layers


class GeneratorSynthesis(nn.Module):
    def __init__(
        self,
        dlatent_size,  # dim w
        device,
        use_specnorm,
        resolution=1024,
        fmap=lambda x: min(int(2**13 / (2.**x)), 512),
        num_channels=3,
        f=None,
        use_wscale=True,
        use_noise=True,
        use_pixel_norm=False,
        use_instanse_norm=True,
        use_styles=True
    ):
        super().__init__()
        N = int(np.log2(resolution))
        w = dlatent_size
        self.num_layers = N * 2 - 2
        self.noise_input = [
            torch.randn([1, 1, 2**(i//2+2), 2**(i//2+2)]).to(device) for i in range(self.num_layers)
        ]
        self.blur = DepthwiseConv2d(f)
        self.channel_shrinkage = Conv2d(
            fmap(N-2), fmap(N), 3, use_wscale=use_wscale
        )
        if use_specnorm:
            self.channel_shrinkage = nn.utils.spectral_norm(
                self.channel_shrinkage)
        self.torgb = Conv2d(fmap(N), num_channels, 1, 1, use_wscale=use_wscale)
        self.adaIn1 = LayerMixtureEpilogue(
            fmap(1), w, use_wscale, use_noise, use_pixel_norm, use_instanse_norm, use_styles
        )
        self.conv1 = Conv2d(
            fmap(1), fmap(1), 3, use_wscale=use_wscale
        )
        if use_specnorm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.adaIn2 = LayerMixtureEpilogue(
            fmap(1), w, use_wscale, use_noise, use_pixel_norm, use_instanse_norm, use_styles
        )
        self.synth = [
            GeneratorBlock(r, use_wscale, use_noise, use_pixel_norm,
                           use_instanse_norm, use_specnorm, self.noise_input, dlatent_size)
            for r in range(3, N + 1)
        ]
        self.synth = nn.ModuleList(self.synth)
        self.const_in = nn.Parameter(torch.ones(1, fmap(1), 4, 4))
        self.bias = nn.Parameter(torch.ones(fmap(1)))

    def forward(self, dlatent):
        w = dlatent
        x = self.const_in.expand(w.size(0), -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.adaIn1(x, self.noise_input[0], w[:, 0])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[1], w[:, 1])
        for layer in self.synth:
            x = layer(x, w)
        x = self.channel_shrinkage(x)
        o = self.torgb(x)
        return o


class StyleGanGenerator(nn.Module):
    def __init__(self, resolution, use_specnorm, device, z_dim=512, w_dim=512, **kwargs):
        super().__init__()
        self.zdim = z_dim
        self.wdim = w_dim
        self.mapper = GeneratorMapper(
            resolution, z_dim, w_dim, **kwargs)
        self.synth = GeneratorSynthesis(
            w_dim, device, use_specnorm, resolution)

    def forward(self, z, phi=.7, cutoff=8):
        w, N = self.mapper(z)
        w = w.unsqueeze(1)
        w = w.expand(-1, int(N), -1)
        coefs = torch.ones([1, N, 1]).to(z.device)
        for i in range(min(cutoff, N)):
            coefs[:, i, :] *= phi
        w = w * coefs
        o = self.synth(w)
        return o


class StyleGanDiscriminator(nn.Module):
    def __init__(
        self,
        resolution,
        use_specnorm,
        f=None,
        num_channels=3,
        b=2**13,
        fm=lambda x, b: min(int(b / (2.**x)), 512)
    ):
        super().__init__()
        N = int(np.log2(resolution))
        self.N = N
        def fmap(x): return fm(x, b)

        def conv(inc, outc, k, s=1, p=0):
            if use_specnorm:
                return nn.utils.spectral_norm(nn.Conv2d(inc, outc, k, stride=s, padding=p))
            else:
                return nn.Conv2d(inc, outc, k, stride=s, padding=p)
        self.fromrgb = conv(num_channels, fmap(N - 1), 1)
        self.blur = DepthwiseConv2d(f)

        self.downlayers = []
        self.convlayers = []
        for r in range(1, N - 1):
            if r < 5:
                self.downlayers.append(nn.AvgPool2d(2))
            else:
                self.downlayers.append(conv(fmap(N-r), fmap(N-r), k=2, s=2))
        for r in range(1, N - 1):
            if r == 1:
                self.convlayers.append(
                    conv(fmap(N-r), fmap(N-r), k=3, p=(1, 1)))
            else:
                self.convlayers.append(
                    conv(fmap(N-r+1), fmap(N-r), k=3, p=(1, 1)))
        self.convlayers = nn.ModuleList(self.convlayers)
        self.downlayers = nn.ModuleList(self.downlayers)
        self.lastconv = conv(fmap(2), fmap(1), 3, p=(1, 1))
        self.fc1 = nn.Linear(b, fmap(0))
        self.fc2 = nn.Linear(fmap(0), 1)

    def forward(self, img):
        x = F.leaky_relu(self.fromrgb(img), 0.2, inplace=True)
        for c, down in zip(self.convlayers, self.downlayers):
            x = F.leaky_relu(c(x), 0.2, inplace=True)
            x = F.leaky_relu(down(self.blur(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.lastconv(x), 0.2, inplace=True)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        return x
