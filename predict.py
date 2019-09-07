import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from models.stylegan import StyleGanGenerator, StyleGanDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random
import logging
import tqdm
import os
from opts import pred_opts
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

logger = logging.getLogger('Style GAN')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = logging.Formatter(
    '[%(module)-10s (%(levelname)-8s) - %(asctime)s] %(message)s'
)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)
random.seed(42)
torch.manual_seed(42)


def normalize(imgs):
    return (imgs - imgs.min()) / (imgs.max() - imgs.min())


def load_generator(opts):
    if os.path.exists(opts.path):
        logger.info("Load Trained Weight")
        state = torch.load(opts.path)
        train_opts = state['opts']
        G = StyleGanGenerator(
            train_opts.imsize,
            train_opts.use_specnorm,
            train_opts.device
        )
        G.load_state_dict(state['G'])
        G.to(train_opts.device)
        opts.device = train_opts.device
        opts.imsize = train_opts.imsize
    else:
        logger.error("Model Files cannot Load")
        raise FileNotFoundError()
    return G


def draw_untruncated_result(G, opts, n=6):
    zs = [torch.randn(1, 512).to(opts.device) for i in range(n**2)]
    imgs = []
    logger.info("Generate Untruncated Images...")
    for z in zs:
        img = G(z).detach().cpu()
        imgs.append(img.numpy()[0].transpose((1, 2, 0)))
    imgs = np.array(imgs)
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    plt.figure(figsize=(n, n), dpi=256)
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    for img, g in zip(imgs, gs):
        ax = plt.subplot(g)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
    plt.savefig(os.path.join(opts.subdir, 'untruncated_result.png'))
    logger.info("Done")


def draw_style_mixing_result(
    G, opts, N_src, N_dst=6,
    style_split=[.25, .5],
    mix=['coarse'] * 2 + ['middle'] * 2 + ['fine'] * 2
):
    import math
    logger.info("Generate Style Mixing Images...")
    logger.info(f"Mixing Setting: {mix}")
    M = G.mapper
    S = G.synth
    src_latents = torch.randn(N_src, 512).to(opts.device)
    dst_latents = torch.randn(N_dst, 512).to(opts.device)
    src_dlatents, src_num_layers = M(src_latents)
    src_dlatents = src_dlatents.unsqueeze(1)
    src_dlatents = src_dlatents.repeat(1, int(src_num_layers), 1)
    dst_dlatents, dst_num_layers = M(dst_latents)
    dst_dlatents = dst_dlatents.unsqueeze(1)
    dst_dlatents = dst_dlatents.repeat(1, int(dst_num_layers), 1)
    coarse = range(0, int(math.ceil(dst_num_layers * style_split[0])))
    middle = range(
        int(math.ceil(dst_num_layers * style_split[0])),
        int(math.ceil(dst_num_layers * style_split[1]))
    )
    fine = range(
        int(math.ceil(dst_num_layers * style_split[0])),
        dst_num_layers
    )
    d = {
        'coarse': coarse,
        'middle': middle,
        'fine': fine
    }
    src_imgs = S(src_dlatents).detach().cpu().numpy()
    dst_imgs = S(dst_dlatents).detach().cpu().numpy()
    plt.figure(figsize=(N_src+1, N_dst+1), dpi=256)
    gs = gridspec.GridSpec(N_dst+1, N_src+1)
    gs.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    for g in gs:
        ax = plt.subplot(g)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplot(gs[0]).imshow(np.zeros((opts.imsize, opts.imsize, 3)))
    for i, src in enumerate(src_imgs):
        ax = plt.subplot(gs[i + 1])
        ax.imshow(normalize(src.transpose((1, 2, 0))))
    for j, dst in enumerate(dst_imgs):
        ax = plt.subplot(gs[(N_src+1)*(j+1)])
        ax.imshow(normalize(dst.transpose((1, 2, 0))))
        row_dlatents = torch.stack([dst_dlatents[j]] * N_src)
        row_dlatents[:, d[mix[j]]] = src_dlatents[:, d[mix[j]]]
        row_images = S(row_dlatents).detach().cpu().numpy()
        for i, img in enumerate(row_images):
            ax = plt.subplot(gs[(N_src+1)*(j+1)+i+1])
            ax.imshow(normalize(img.transpose((1, 2, 0))))
    plt.savefig(os.path.join(opts.subdir, 'mixing_result.png'))
    logger.info("Done")


def draw_noise_detail_result(G, opts, N_sample=5, valiation=4):
    logger.info("Generate Noise Detail Images...")
    plt.figure(figsize=(valiation, N_sample), dpi=256)
    gs = gridspec.GridSpec(N_sample, valiation)
    gs.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    for g in gs:
        ax = plt.subplot(g)
        ax.set_xticks([])
        ax.set_yticks([])
    for row in range(N_sample):
        latents = torch.randn(1, 512).to(opts.device).repeat((valiation, 1))
        images = G(latents).detach().cpu().numpy()
        for col, img in enumerate(images):
            ax = plt.subplot(gs[row*valiation + col])
            ax.imshow(normalize(img.transpose((1, 2, 0))))
    plt.savefig(os.path.join(opts.subdir, 'noise_result.png'))
    logger.info("Done")


def draw_truncated_result(G, opts, n=6, psi=.7, off=8):
    zs = [torch.randn(1, 512).to(opts.device) for i in range(n**2)]
    imgs = []
    logger.info("Generate Truncate Trick Images...")
    M = G.mapper
    S = G.synth
    for z in zs:
        w, N = M(z)
        w.unsqueeze(1)
        w.expand(w.size(0), int(N), -1)
        coefs = torch.ones((1, N, 1)).to(opts.device)
        for i in range(int(N)):
            if i < off:
                coefs[:, i, :] *= psi
        w = w + coefs
        img = S(w).detach().cpu().numpy()[0].transpose((1, 2, 0))
        imgs.append(img)
    imgs = np.array(imgs)
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    plt.figure(figsize=(n, n), dpi=256)
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    for img, g in zip(imgs, gs):
        ax = plt.subplot(g)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
    plt.savefig(os.path.join(opts.subdir, 'truncated_result.png'))
    logger.info("Done")


if __name__ == '__main__':
    opts = pred_opts.PredictPerser()
    G = load_generator(opts)
    if opts.draw_pred:
        draw_untruncated_result(G, opts)
    if opts.draw_mix:
        draw_style_mixing_result(G, opts, 5)
    if opts.draw_noise:
        draw_noise_detail_result(G, opts)
    if opts.draw_trunc:
        draw_truncated_result(G, opts)
