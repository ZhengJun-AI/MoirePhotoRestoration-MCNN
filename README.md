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

## Dataset

| psnr distribution | \<12 | 12~14 | 14~17 | 17~20 | 20~22 | 22~24 | \>24  |
| :---------------: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
|   training set    |  72  | 2318  | 29816 | 37089 | 21195 | 15102 | 12856 |
|    testing set    |  8   |  227  | 2951  | 3809  | 2069  | 1463  | 1324  |
|       total       |  80  | 2545  | 32767 | 40898 | 23264 | 16565 | 14180 |

We can see that low quality image pairs whose PSNR is lower than 12 still exist in the dataset, which is against the author's declaration.