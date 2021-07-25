import torch.utils.data as data
from torchvision import transforms
from random import sample
from PIL import Image, ImageFile
import os
from tqdm import tqdm


class MoirePic(data.Dataset):
    def __init__(self, rootX, rootY, training=True):
        self.picX = [os.path.join(rootX, img) for img in os.listdir(rootX)]
        self.picY = [os.path.join(rootY, img) for img in os.listdir(rootY)]
        self.picX.sort()
        self.picY.sort()
        self.pics = list(zip(self.picX, self.picY))
        # self.pics = self.pics[:400]
        self.Len = len(self.pics)

        if not training:
            self.pics = sample(self.pics, self.Len // 10)
            self.Len = len(self.pics)

    def __getitem__(self, index):
        tf = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])

        path_pair = self.pics[index]
        imgX, imgY = Image.open(path_pair[0]), Image.open(path_pair[1])
        return tf(imgX), tf(imgY)

    def __len__(self):
        return self.Len


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.fill_(0)


if __name__=='__main__':
    root = '/data_new/zxbsmk/moire/trainData'
    # you need to clean the training set
    input_path = os.path.join(root, 'source')
    gt_path = os.path.join(root, 'target')
    input_imgs = [os.path.join(input_path, img) for img in os.listdir(input_path)]
    gt_imgs = [os.path.join(gt_path, img) for img in os.listdir(gt_path)]
    input_imgs.sort()
    gt_imgs.sort()

    cot = 0
    loop = tqdm(enumerate(input_imgs), total=len(input_imgs), leave=False)
    for idx, img in loop:
        with open(img, "rb") as f:
            ImPar=ImageFile.Parser()
            chunk = f.read(2048)
            count=2048
            while chunk != "":
                ImPar.feed(chunk)
                if ImPar.image:
                    break
                chunk = f.read(2048)
                count+=2048
            M, N = ImPar.image.size[0], ImPar.image.size[1]

        if M < 260 or N < 260:
            os.remove(input_imgs[idx])
            os.remove(gt_imgs[idx])
            cot += 1
        
        loop.set_postfix(unfit_imgs=cot)

    print("Done! Get %d unfit images." % cot)