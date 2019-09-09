# Pytorch版StyleGAN

[![CC](https://img.shields.io/badge/license-CC_BY--NC_4.0-green.svg?style=flat)](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt)

このリポジトリは[StyleGAN](http://stylegan.xyz/paper)のPytorchによる追実装です．主部分は，[このリポジトリ](https://github.com/tomguluson92/StyleGAN_PyTorch)を参照しています．元リポジトリ同様，[torchvision_sunner](https://github.com/SunnerLi/Torchvision_sunner)による画像の読み込みを使用しています．

## システム要件

CPUでも動作可能ですが非常に時間がかかります．

開発時のスペックは以下の通りです

- OS: Ubuntu 16.04.6 LTS
- RAM: 128GB
- CPU: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
- GPU: Tesla P100-PCIE-16GB
- CUDA: 10.0
- CUDNN: 7.5
- PyTorch: 1.1.0
- Torchvision: 0.2.2
- BatchSize: 1024x1024: 2, 256x256: 9

## 学習

```bash
# 1. 適当なフォルダ下に訓練用画像を入れてください

# 2. (GPUの場合)CUDA対応のPyTorch/Torch Visionが入っていることを確認してください
#    Dockerの場合は対応デバイスのPyTorchイメージより環境を作成してください

# 3. pipで依存パッケージをインストールします．
$ pip install numpy matplotlib tqdm scikit-image

# 4. 訓練
$ python train.py -i /path/to/training/images
```

### オプション引数

```shell
usage: train.py [-h] -i DIRECTORY [-o DIRECTORY] [-n NUMBER] [-b INTEGER]
                [-r [PATH|None]] [-s RESOLUTION] [--no-gpu] [--use-specnorm]
                [--critic-iters INTEGER] [--show-interval INTEGER]
                [--r1gamma NON_NEGATIVE] [--r2gamma NON_NEGATIVE]
                [--lr GEN_LR DIS_LR] [--lr-decay GEN_LR DIS_LR]
                [--betas BETA1 BETA2]

optional arguments:
  -h, --help            show this help message and exit
  -i DIRECTORY, --input DIRECTORY
                        Directory with input images (default: None)
  -o DIRECTORY, --output DIRECTORY
                        Output destination (default: ./result/)
  -n NUMBER, --epochs NUMBER
                        Number of training iterations (default: 1000)
  -b INTEGER, --batch-size INTEGER
                        Batch size (default: 2)
  -r [PATH|None], --resume [PATH|None]
                        Directory where resume data exists (None: Not
                        specified) (default: None)
  -s RESOLUTION, --imsize RESOLUTION
                        Image resolution (default: 1024)
  --no-gpu              Use CPU only
  --use-specnorm        Whether to impose a Spectral Norm on the model
  --critic-iters INTEGER
                        Ratio of discriminator training times to one generator
                        training (default: 5)
  --show-interval INTEGER
                        Progress output interval (default: 250)
  --r1gamma NON_NEGATIVE
                        Gradient penalty coefficient for discrimination result
                        of real image (default: 0.0)
  --r2gamma NON_NEGATIVE
                        Gradient penalty coefficient for the discrimination
                        result of generated image (default: 0.0)
  --lr GEN_LR DIS_LR    Learning rate (Generator,Discriminator) (default:
                        [0.0002434, 0.0002434])
  --lr-decay GEN_LR DIS_LR
                        Learning rate decay rate (Generator,Discriminator)
                        (default: [0.9, 0.9])
  --betas BETA1 BETA2   Adam Optimizer hyperparameters (default: [0.0, 0.99])
```

**説明**

|引数(短)|引数(長)|型|説明|
|-:|-:|-:|:-|
|`-i`|`--input`|`string`|(必須)入力画像ディレクトリへのパス|
|`-o`|`--output`|`string`|出力先(デフォルト: `./result`)|
|`-n`|`--epochs`|`int`|エポック数(デフォルト: `1000`)|
|`-b`|`--batch-size`|`int`|バッチサイズ(デフォルト: `2`)|
|`-r`|`--resume`|`string or None`|途中から再開する際にモデルデータの保存されたパスを指定.(デフォルト: `None`)|
|`-s`|`--imsize`|`int`|画像サイズ(`32~1024`の2の累乗のみ指定可)(デフォルト: `1024`)|
||`--no-gpu`|`(switch)`|GPU を使用しないようにしたい場合に指定する|
||`--use-specnorm`|`(switch)`|モデルにSpectral Normalizationを課す場合に指定する|
||`--critic-iters`|`int`|Generator1回の訓練に対するDiscriminatorの訓練回数の比(デフォルト: `5`)|
||`--show-interval`|`int`|進捗の出力間隔(デフォルト: `250`)|
||`--r1gamma`|`float(>0)`|本物画像と識別した場合に対する勾配への罰則係数(デフォルト: `0`)|
||`--r2gamma`|`float(>0)`|生成画像と識別した場合に対する勾配への罰則係数(デフォルト: `0`)|
||`--lr`|`float float`|学習率(デフォルト: `2.434e-4 2.434e-4`)|
||`--lr-decay`|`float float`|1Epochあたりの学習率の減衰率(デフォルト: `0.9 0.9`)|
||`--betas`|`float float`|Adam Optimizerのハイパーパラメータ(デフォルト: `0 0.99`)|

## 生成

```bash
# 1. モデルのセーブファイルが存在することを確認してください
#    デフォルトでは`./result/models/latest.pth`です

# 2. 生成
$ python predict.py -f ./result/models/latest.pth
```

### オプション引数

```shell
usage: predict.py [-h] [-f PTH_FILE] [-o PATH] [--no-gpu]

optional arguments:
  -h, --help            show this help message and exit
  -f PTH_FILE, --model-file PTH_FILE
                        Path to trained model (default:
                        ./result/models/latest.pth)
  -o PATH, --output-path PATH
                        Output root path (default: ./generate/)
  --no-gpu              Use CPU only
```

**説明**

|引数(短)|引数(長)|型|説明|
|-:|-:|-:|:-|
|`-f`|`--model-file`|`string`|モデルへのパス(デフォルト: `./result/models/latest.pth`)|
|`-o`|`--output`|`string`|出力先(デフォルト: `./generate`)|
||`--no-gpu`|`(switch)`|GPU を使用しないようにしたい場合に指定する|
