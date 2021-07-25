# MoirePhotoRestoration-MCNN

This is an unofficial reproduction of paper *Moir´e Photo Restoration Using Multiresolution Convolutional Neural Networks*.(PyTorch)

First of all, you need to prepare the whole dataset of this paper, which is around 100G.\
dataset download link : https://drive.google.com/drive/folders/109cAIZ0ffKLt34P7hOMKUO14j3gww2UC

## Requirements

* torch >= 1.6.0
* torchvision >= 0.7.0
* pillow >= 7.2.0
* GPU >= 3G

## Training

Before starting to train the model, you need to run a script to clean the training set as shown below.\
All hyper-parameters follow the instructions of the paper, so you don't need to change them.W

However, you should change the path of datasets to match your local environment.

```bash
python utils.py
python train.py --dataset /data_new/zxbsmk/moire/trainData --save ./model
```

## Testing

Get PSNR of the testing set.

```bash
python test.py
```

