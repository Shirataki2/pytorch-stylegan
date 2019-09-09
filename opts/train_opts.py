import torch
import datetime
import os
import argparse
import sys
import logging
from models.stylegan import StyleGanGenerator, StyleGanDiscriminator


logger = logging.getLogger('Style GAN')


class TrainParser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        # 入力画像のあるディレクトリ
        parser.add_argument(
            '-i', '--input',
            metavar='DIRECTORY',
            help='Directory with input images (default: %(default)s)',
            type=str,
            required=True
        )
        # 出力先
        parser.add_argument(
            '-o', '--output',
            metavar='DIRECTORY',
            help='Output destination (default: %(default)s)',
            type=str,
            default='./result/'
        )
        # 訓練反復回数
        parser.add_argument(
            '-n', '--epochs',
            metavar='NUMBER',
            help='Number of training iterations (default: %(default)s)',
            type=int,
            default=1000
        )
        # バッチサイズ
        parser.add_argument(
            '-b', '--batch-size',
            metavar='INTEGER',
            help='Batch size (default: %(default)s)',
            type=int,
            default=2
        )
        # 再開データの存在するディレクトリ(Noneで指定なし)
        parser.add_argument(
            '-r', '--resume',
            metavar='[PATH|None]',
            help='Directory where resume data exists (None: Not specified) (default: %(default)s)',
            default=None
        )
        # 画像の解像度
        parser.add_argument(
            '-s', '--imsize',
            metavar='RESOLUTION',
            help='Image resolution (default: %(default)s)',
            type=int,
            default=1024
        )
        # GPUを使うかどうか
        parser.add_argument(
            '--no-gpu',
            help='Use CPU only',
            action='store_true',
        )
        # モデルにSpectral Normを課すかどうか
        parser.add_argument(
            '--use-specnorm',
            help='Whether to impose a Spectral Norm on the model',
            action='store_true',
        )
        # 生成器1回の訓練に対する弁別器の訓練回数の比
        parser.add_argument(
            '--critic-iters',
            metavar='INTEGER',
            help='Ratio of discriminator training times to one generator training (default: %(default)s)',
            type=int,
            default=5
        )
        # 進捗の出力間隔
        parser.add_argument(
            '--show-interval',
            metavar='INTEGER',
            help='Progress output interval (default: %(default)s)',
            type=int,
            default=250
        )
        # 本物画像の弁別結果に対する勾配罰則係数
        parser.add_argument(
            '--r1gamma',
            metavar='NON_NEGATIVE',
            help='Gradient penalty coefficient for discrimination result of real image (default: %(default)s)',
            type=float,
            default=0.
        )
        # 生成画像の弁別結果に対する勾配罰則係数
        parser.add_argument(
            '--r2gamma',
            metavar='NON_NEGATIVE',
            help='Gradient penalty coefficient for the discrimination result of generated image (default: %(default)s)',
            type=float,
            default=0.
        )
        # 学習率(Generator,Discriminator)
        parser.add_argument(
            '--lr',
            metavar=('GEN_LR', 'DIS_LR'),
            nargs=2, type=float,
            help='Learning rate (Generator,Discriminator) (default: %(default)s)',
            default=[2.434e-4, 2.434e-4]
        )
        # 学習率の減衰率(Generator,Discriminator)
        parser.add_argument(
            '--lr-decay',
            metavar=('GEN_LR', 'DIS_LR'),
            help='Learning rate decay rate (Generator,Discriminator) (default: %(default)s)',
            nargs=2, type=float,
            default=[.9, .9]
        )
        # Adam Optimizerのハイパーパラメータ
        parser.add_argument(
            '--betas', nargs=2,
            metavar=('BETA1', 'BETA2'),
            help='Adam Optimizer hyperparameters (default: %(default)s)',
            type=float, default=[.0, .99]
        )
        self.opts = parser.parse_args()
        self.parse()

    def parse(self):
        opts = self.opts
        load_state = False
        state = None
        if opts.resume is not None:
            try:
                path = opts.resume
                state = torch.load(opts.resume)
                logger.info(
                    f"Resume Path: {opts.resume} (Start epoch: {state['start_epoch']})"
                )
                opts = state['opts']
                opts.resume = path
                load_state = True
            except:
                logger.warn(f"Failed to load resume model")
                logger.info(f"Resume Path: None (Train from scratch)")
        else:
            logger.info(f"Resume Path: None (Train from scratch)")
        self.use_gpu = not opts.no_gpu
        self.lr = opts.lr
        self.lr_decay = opts.lr_decay
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Use {self.device.upper()}")
        self.input = opts.input
        logger.info(f"Input  Folder: {self.input}")
        self.output = opts.output
        logger.info(f"Output Folder: {self.output}")
        self.batch_size = opts.batch_size
        logger.info(f"Batch Size: {self.batch_size}")
        os.makedirs(self.output, exist_ok=True)
        subdir = self.output
        os.makedirs(subdir, exist_ok=True)
        os.makedirs(os.path.join(subdir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(subdir, 'images', 'fixed'), exist_ok=True)
        os.makedirs(os.path.join(subdir, 'images', 'normal'), exist_ok=True)
        os.makedirs(os.path.join(subdir, 'models'), exist_ok=True)
        self.epochs = opts.epochs
        logger.info(f"Train Epochs: {self.epochs}")
        self.critic_iters = opts.critic_iters
        logger.info(f"Critic Iters: {self.critic_iters}")
        self.use_specnorm = opts.use_specnorm
        logger.info(f"Use Sepctral Norm: {self.use_specnorm}")
        self.resume = opts.resume
        lr = self.lr
        lr_decay = self.lr_decay
        self.g_lr = lr[0]
        self.d_lr = lr[1]
        self.g_lrdecay = lr_decay[0]
        self.d_lrdecay = lr_decay[1]
        logger.info(
            f"Learning Rate: Generator    : {self.g_lr} (Decay factor: {self.g_lrdecay})"
        )
        logger.info(
            f"Learning Rate: Discriminator: {self.d_lr} (Decay factor: {self.d_lrdecay})"
        )
        self.betas = opts.betas
        logger.info(
            f"Use Adam Optim (beta1: {self.betas[0]}, beta2: {self.betas[1]})"
        )
        self.imsize = opts.imsize
        supported_resolution = [2 ** i for i in range(6, 11)]
        if self.imsize not in supported_resolution:
            logger.critical(f"Unsupported Resolution: {self.imsize}")
            logger.info(f"support only: {supported_resolution}")
            exit(1)
        logger.info(f"Resolution: {self.imsize}")
        self.r1gamma = opts.r1gamma
        self.r2gamma = opts.r2gamma
        self.show_interval = opts.show_interval
        self.G = StyleGanGenerator(self.imsize, self.use_specnorm, self.device)
        self.D = StyleGanDiscriminator(self.imsize, self.use_specnorm)
        self.start_epoch = 1
        if load_state:
            self.G.load_state_dict(state['G'])
            self.D.load_state_dict(state['D'])
            self.start_epoch = state['start_epoch']
