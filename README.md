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

|引数(短)|引数(長)|型|説明|
|-:|-:|-:|:-|
|`-i`|`--input`|`string`|(必須)入力画像ディレクトリへのパス|
|`-o`|`--output`|`string`|出力先(デフォルト: `./result`)|
|`-n`|`--epochs`|`int`|エポック数(デフォルト: `1000`)|
|`-b`|`--batch-size`|`int`|バッチサイズ(デフォルト: 2)|
|`-r`|`--resume`|`string or None`|途中から再開する際にモデルデータの保存されたパスを指定.(デフォルト: `None`)|
||`--no-gpu`|`(switch)`|GPU を使用しないようにしたい場合に指定する|
||`--use-specnorm`|`(switch)`|モデルにSpectral Normalizationを課す場合に指定する|
||`--critic-iters`|`int`|Generator1回の訓練に対するDiscriminatorの訓練回数の比(デフォルト: `5`)|
||`--show-interval`|`int`|進捗の出力間隔(デフォルト: `250`)|
||`--r1gamma`|`float(>0)`|本物画像と識別した場合に対する勾配への罰則係数(デフォルト: `10`)|
||`--r2gamma`|`float(>0)`|生成画像と識別した場合に対する勾配への罰則係数(デフォルト: `0`)|
||`--lr`|`float float`|学習率(デフォルト: `2.434e-5 2.434e-5`)|
||`--lr-decay`|`float float`|1Epochあたりの学習率の減衰率(デフォルト: `0.99 0.99`)|
||`--betas`|`float float`|Adam Optimizerのハイパーパラメータ(デフォルト: `0.5 0.99`)|

## 生成

```bash
# 1. モデルのセーブファイルが存在することを確認してください
#    デフォルトでは`./result/models/latest.pth`です

# 2. 生成
$ python predict.py -f ./result/models/latest.pth
```

### オプション引数

|引数(短)|引数(長)|型|説明|
|-:|-:|-:|:-|
|`-f`|`--model-file`|`string`|モデルへのパス(デフォルト: `./result/models/latest.pth`)|
|`-o`|`--output`|`string`|出力先(デフォルト: `./generate`)|
||`--no-gpu`|`(switch)`|GPU を使用しないようにしたい場合に指定する|
