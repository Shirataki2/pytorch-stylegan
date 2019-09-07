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
        parser.add_argument(
            '--no-gpu',
            help='Use CPU only',
            action='store_true',
        )
        self.opts = parser.parse_args()
        self.parse()

    def parse(self):
        opts = self.opts
        self.path = opts.model_file
        self.use_gpu = not opts.no_gpu
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Use {self.device.upper()}")
        logger.info(f"Model File  : {self.path}")
        self.output = opts.output_path
        logger.info(f"Output Path : {self.output}")
        os.makedirs(self.output, exist_ok=True)
        self.subdir = os.path.join(
            self.output,
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        os.makedirs(self.subdir, exist_ok=True)
