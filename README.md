# 🚀 papersDL

Implementation of scientific papers on deep learning, mostly developed using Tensorflow.

## 📄 Index

### 👀 Computer Vision

| Paper                                                                 | Code                                |
|-----------------------------------------------------------------------|---------------------------------------|
| [Truly Shift-Invariant Convolutional Neural Networks](https://arxiv.org/pdf/2011.14214) | [APS.py](cv/APS.py)                   |
| [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070) | [BiFPN.py](cv/BiFPN.py)               |
| [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/pdf/1904.11486) | [BlurPool.py](cv/BlurPool.py)         |
| [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521) | [CBAM.py](cv/CBAM.py)                 |
| [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545)          | [ConvNextResidualBlock.py](cv/ConvNextResidualBlock.py) |
| [All the Attention You Need: Global-Local, Spatial-Channel Attention for Image Retrieval](https://arxiv.org/pdf/2107.08000) | [GLAM.py](cv/GLAM.py)                 |
| [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842)    | [googlelenet.py](cv/googlelenet.py)   |
| [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | [lenet.py](cv/lenet.py)               |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) | [resnet.py](cv/resnet.py)             |
| [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)  | [SE.py](cv/SE.py)                     |
| [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729) | [SPP.py](cv/SPP.py)                   |
| [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899) | [CutMix.py](cv/CutMix.py)                   |
| [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412) | [MixUp.py](cv/MixUp.py)                   |

### 📉 Loss functions

| Paper                                                                 | Code                                |
|-----------------------------------------------------------------------|---------------------------------------|
| [Contractive Auto-Encoders: Explicit Invariance During Feature Extraction](http://www.icml-2011.org/papers/455_icmlpaper.pdf) | [ContractiveLoss.py](losses/ContractiveLoss.py)                   |
| [Supervised Contrastive Learning](https://arxiv.org/pdf/2004.11362) | [SupervisedContrastiveLoss.py](losses/SupervisedContrastiveLoss.py)               |

### 🪛 Optimizers

| Paper                                                                 | Code                                |
|-----------------------------------------------------------------------|---------------------------------------|
| [AUTOCLIP: ADAPTIVE GRADIENT CLIPPING FOR SOURCE SEPARATION NETWORKS](https://arxiv.org/pdf/2007.14469) | [AGC.py](optimizers/AGC.py)                   |
| [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/pdf/2004.01461) | [GCAdamW.py](optimizers/GCAdamW.py)               |

### ➿ Learning rate schedulers

| Paper                                                                 | Code                                |
|-----------------------------------------------------------------------|---------------------------------------|
| [SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS](https://arxiv.org/pdf/1608.03983) | [WarmUpCosine.py](schedulers/WarmUpCosine.py)                   |
