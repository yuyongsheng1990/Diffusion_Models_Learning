# -*- coding: utf-8 -*-
# @Time : 2024/6/17 22:36
# @Author : yysgz
# @File : dataset.py
# @Project : DDPM_model_2
# @Description : https://github.com/SingleZombie/DL-Demos/tree/master/dldemos/ddpm

import torchvision  # 提供了MNIST数据接口，调用接口就可以生成MNIST dataset。
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor  # compose将多个图像变换组合在一起；ToTensor()用于将PIL img或numpy 数组转换为tensor。

import os

def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print('length of MNIST', len(mnist))
    id = 4
    img, label = mnist[id]
    print(img)
    print(label)

    img.save('work_dirs/tmp.jpg')
    tensor = ToTensor()(img)
    print(tensor.shape)  # single channel picture
    print(tensor.max())  # value range 0~1
    print(tensor.min())

def get_downloader(batch_size: int):  # 创建 Dataloader
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])  # 因为DDPM会把图像和DDPM联系起来，所以我们希望picture取值范围是[-1,1]
    dataset = torchvision.datasets.MNIST(root='./data/mnist', transform=transform)
    # 在处理大型数据集时，dataloader提供了非常高效的数据加载方式：batch批量处理、shuffle打乱、num_workers并行加载、transformers数据预处理(归一化、裁剪、翻转)、与DataSet集成使用。
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_img_shape():
    return (1, 28, 28)

if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)
    download_dataset()
