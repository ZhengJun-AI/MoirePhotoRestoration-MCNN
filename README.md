# MoirePhotoRestoration-MCNN

摩尔纹消除领域经典论文 *Moir´e Photo Restoration Using Multiresolution Convolutional Neural Networks* 复现

文件共有两个：train.py 和 moire.ipynb\
数据集下载地址：https://drive.google.com/drive/folders/109cAIZ0ffKLt34P7hOMKUO14j3gww2UC \
论文分析博文地址：https://www.zxbsmk.top/index.php/2020/08/21/2018%e5%b9%b4-demoire-%e7%bb%8f%e5%85%b8%e8%ae%ba%e6%96%87%e5%a4%8d%e7%8e%b0/

（以下操作都需要替换文件中所包含的数据集绝对路径）\
在运行 train.py 之前，请先运行 moire.ipynb 中的**实验数据清理**部分，确保剩下的图片都能被成功裁切\
并且最好在 notebook 中成功运行整个文件后，确保自己理解并能够自如地调整各个参数后，再去改写并运行 train.py

（对于Windows用户，推荐安装 Terminal ）\
确定 train.py 无误后，可以通过 nohup 命令后台运行，并且把运行进度输出到文件中，确保自己能够实时了解训练进度\
以下是我个人使用的命令，作为示范\
我将运行进度输出到 res.txt 文件中，只要使用 cat 命令就能实时查看
```
nohup python3 train.py 1>res.txt &
cat res.txt
```
