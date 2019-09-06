import torch
import datetime
import os
import argparse
import sys
import logging


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
            default='./data/danbooru'
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
            '--epochs',
            metavar='NUMBER',
            help='Number of training iterations (default: %(default)s)',
            type=int,
            default=1000
        )
        # バッチサイズ
        parser.add_argument(
            '--batch-size',
            metavar='INTEGER',
            help='Batch size (default: %(default)s)',
            type=int,
            default=2
        )
        # 再開データの存在するディレクトリ(Noneで指定なし)
        parser.add_argument(
            '--resume',
            metavar='[PATH|None]',
            help='Directory where resume data exists (None: Not specified) (default: %(default)s)',
            default=None
        )
        # 画像の解像度
        parser.add_argument(
            '-s', '--image-size',
            metavar='RESOLUTION',
            help='Image resolution (default: %(default)s)',
            type=int,
            default=1024
        )
        # GPUを使うかどうか
        parser.add_argument(
            '--use-gpu',
            metavar='BOOLEAN',
            help='Whether to use GPU (default: %(default)s)',
            type=bool,
            default=True
        )
        # モデルにSpectral Normを課すかどうか
        parser.add_argument(
            '--use-specnorm',
            metavar='BOOLEAN',
            help='Whether to impose a Spectral Norm on the model (default: %(default)s)',
            type=bool,
            default=False
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
            '--r1-gamma',
            metavar='NON_NEGATIVE',
            help='Gradient penalty coefficient for discrimination result of real image (default: %(default)s)',
            type=float,
            default=10.
        )
        # 生成画像の弁別結果に対する勾配罰則係数
        parser.add_argument(
            '--r2-gamma',
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
            default=[2.434e-5, 2.434e-5]
        )
        # 学習率の減衰率(Generator,Discriminator)
        parser.add_argument(
            '--lr-decay',
            metavar=('GEN_LR', 'DIS_LR'),
            help='Learning rate decay rate (Generator,Discriminator) (default: %(default)s)',
            nargs=2, type=float,
            default=[.995, .995]
        )
        # Adam Optimizerのハイパーパラメータ
        parser.add_argument(
            '--betas', nargs=2,
            metavar=('BETA1', 'BETA2'),
            help='Adam Optimizer hyperparameters (default: %(default)s)',
            type=float, default=[.5, .99]
        )
        self.opts = parser.parse_args()
        self.parse()

    def parse(self):
        opts = self.opts
        self.device = 'cuda' if opts.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Use {self.device}")
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
        if self.resume:
            logger.info(f"Resume Path: {self.resume}")
        else:
            logger.info(f"Resume Path  None (Train from scratch)")
        lr = opts.lr
        lr_decay = opts.lr_decay
        if len(lr) >= 3:
            logger.warn(
                "The length of option 'lr' is invalid. The length should be 2 or 3"
            )
        if len(lr) == 1:
            lr = lr * 2
        self.g_lr = lr[0]
        self.d_lr = lr[1]
        if len(lr_decay) >= 3:
            logger.warn(
                "The length of option 'lr_decay' is invalid. The length should be 2 or 3"
            )
        if len(lr_decay) == 1:
            lr_decay = lr_decay * 2
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
        self.imsize = opts.image_size
        supported_resolution = [2 ** i for i in range(6, 11)]
        if self.imsize not in supported_resolution:
            logger.critical(f"Unsupported Resolution: {self.imsize}")
            logger.info(f"support only: {supported_resolution}")
            exit(1)
        logger.info(f"Resolution: {self.imsize}")
        self.r1gamma = opts.r1_gamma
        self.r2gamma = opts.r2_gamma
        self.show_interval = opts.show_interval
