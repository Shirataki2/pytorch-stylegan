import torch
import datetime
import os
import argparse
import sys
import logging

logger = logging.getLogger('Style GAN')


class PredictPerser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        # 訓練済みモデルへのパス
        parser.add_argument(
            '-f', '--model-file',
            metavar='PTH_FILE',
            help='Path to trained model (default: %(default)s)',
            type=str,
            default='./result/models/latest.pth'
        )
        # 出力ルートパス
        parser.add_argument(
            '-o', '--output-path',
            metavar='PATH',
            help='Output root path (default: %(default)s)',
            type=str,
            default='./generate/'
        )
        # 潜在変数に対し処理をなにも行わずに画像を生成する処理をするかどうか
        parser.add_argument(
            '-p', '--pred',
            metavar='BOOLEAN',
            help='Whether to generate an image without performing any processing on latent variables (default: %(default)s)',
            type=bool,
            default=True
        )
        # 画像間を補完するように画像を生成する処理をするかどうか
        parser.add_argument(
            '-m', '--mixture-pred',
            metavar='BOOLEAN',
            help='Whether to generate images to complement between images (default: %(default)s)',
            type=bool,
            default=True
        )
        # ノイズの付加による生成画像の変化の様子を出力する処理を行うかどうか
        parser.add_argument(
            '-n', '--noise-pred',
            metavar='BOOLEAN',
            help='Whether to perform processing to output the change in the generated image due to the addition of noise (default: %(default)s)',
            type=bool,
            default=True
        )
        # 袖を打ち切る正規表現トリック(Truncation Trick)を用いた画像を生成する処理を行うかどうか
        parser.add_argument(
            '-t', '--trunc-pred',
            metavar='BOOLEAN',
            help='Whether to perform image generation using Truncation Trick (default: %(default)s)',
            type=bool,
            default=True
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
        self.opts = parser.parse_args()
        self.parse()

    def parse(self):
        opts = self.opts
        self.draw_pred = opts.pred
        self.draw_mix = opts.mixture_pred
        self.draw_noise = opts.noise_pred
        self.draw_trunc = opts.trunc_pred
        self.path = opts.model_file
        logger.info(f"Model File  : {self.path}")
        self.output = opts.output_path
        logger.info(f"Output Path : {self.output}")
        os.makedirs(self.output, exist_ok=True)
        self.subdir = os.path.join(
            self.output,
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        os.makedirs(self.subdir, exist_ok=True)
        self.imsize = opts.image_size
        supported_resolution = [2 ** i for i in range(6, 11)]
        if self.imsize not in supported_resolution:
            logger.critical(f"Unsupported Resolution: {self.imsize}")
            logger.info(f"support only: {supported_resolution}")
            exit(1)
        logger.info(f"Resolution: {self.imsize}")
        self.use_specnorm = opts.use_specnorm
        logger.info(f"Use Sepctral Norm: {self.use_specnorm}")
        self.device = 'cuda' if opts.use_gpu and torch.cuda.is_available() else 'cpu'