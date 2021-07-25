import numpy as np
import os
import math
import torch
from tqdm import tqdm
from utils import MoirePic
from torch.utils.data import DataLoader


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10(1 / mse)


if __name__ == '__main__':
    root = '/data_new/zxbsmk/moire/testData'
    dataset = MoirePic(os.path.join(root, 'source'),
                       os.path.join(root, 'target'))
    test_loader = DataLoader(dataset=dataset, batch_size=4, drop_last=False)
    # you can change batchsize to get a faster speed
    model = torch.load('./model/moire_best.pth')
    model.eval()

    psnr_all, count = 0.0, 0
    loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
    for idx, (data, target) in loop:
        data, target = data.cuda(), target.numpy()

        with torch.no_grad():
            output = model(data).cpu().numpy()

        for i in range(target.shape[0]):
            psnr_all += psnr(output[i], target[i])

        count += target.shape[0]
        loop.set_postfix(average_psnr=psnr_all / count)

    print(f'testing dataset PSNR: {psnr_all / count}')
