import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from stylegan import StyleGanGenerator, StyleGanDiscriminator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random
import logging
import tqdm
import os
from opts import pred_opts

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


def load_generator(opts):
    G = StyleGanGenerator(opts.imsize, opts.use_specnorm, opts.device)
    if os.path.exists(opts.path):
        logger.info("Load Trained Weight")
        state = torch.load(opts.path)
        G.load_state_dict(state['G'])
        start_epoch = state['start_epoch']
    else:
        logger.error("Model Files cannot Load")
        raise FileNotFoundError()
    G.to(opts.device)
    return G


def draw_untruncated_result(G, opts, n=6):
    zs = [torch.randn(1, 512).to(opts.device) for i in range(n**2)]
    imgs = []
    for z in tqdm.tqdm(zs):
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


def draw_style_mixing_result(G, opts, N_src, N_dst):
    M = G.mapper
    S = G.synth
    src_latents = torch.randn(N_src, 512).to(opts.device)
    dst_latents = torch.randn(N_dst, 512).to(opts.device)
    src_dlatents, src_num_layers = M(src_latents)
    src_dlatents = src_dlatents.unsqueeze(1)
    src_dlatents = src_dlatents.expand(-1, int(src_num_layers), -1)
    dst_dlatents, dst_num_layers = M(dst_latents)
    dst_dlatents = dst_dlatents.unsqueeze(1)
    dst_dlatents = dst_dlatents.expand(-1, int(dst_num_layers), -1)
    src_imgs = S(src_dlatents).detach().cpu().numpy()
    dst_imgs = S(dst_dlatents).detach().cpu().numpy()
    plt.figure(figsize=(N_src+1, N_dst+1), dpi=256)
    gs = gridspec.GridSpec(N_src+1, N_dst+1)
    gs.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    for g in gs:
        ax = plt.subplot(g)
        ax.set_xticks([])
        ax.set_yticks([])
    for i, src in enumerate(src_imgs):
        ax = plt.subplot(gs[i + 1])
        ax.imshow(src)
    for j, dst in enumerate(dst_imgs):
        ax = plt.subplot(gs[(N_dst+1)*(j+1)])
        ax.imshow(dst)
    plt.savefig(os.path.join(opts.subdir, 'mixing_result.png'))


if __name__ == '__main__':
    opts = pred_opts.PredictPerser()
    G = load_generator(opts)
    if opts.draw_pred:
        draw_untruncated_result(G, opts)
    draw_style_mixing_result(G, opts, 5, 5)
