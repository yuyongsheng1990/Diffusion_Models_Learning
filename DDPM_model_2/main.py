# -*- coding: utf-8 -*-
# @Time : 2024/6/18 15:34
# @Author : yysgz
# @File : main.py
# @Project : DDPM_model_2
# @Description : training

import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from dataset import get_downloader, get_img_shape
from ddpm import DDPM
from network import build_network, convnet_big_cfg, convnet_medium_cfg, convnet_small_cfg, unet_1_cfg, unet_res_cfg

batch_size = 512
n_epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(ddpm: DDPM, net, device=device, ckpt_path='model.pth'):
    print('batch_size:', batch_size)
    n_steps = ddpm.n_steps
    dataloader = get_downloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)  # 随机生成一个range在(0, n_steps)的整数型t tensor, shape为(current_batch_size,)
            eps = torch.randn_like(x).to(device)  # 生成一个前向过程中加入的随机噪声noise
            x_t = ddpm.sample_forward(x, t, eps)  # x_t = sqrt(α_t_bar) * x + sqrt(1-α_t_bar) * ε
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))  # 估计反向过程中需要剔除的noise_θ at each step.
            loss = loss_fn(eps_theta, eps)  # 优化的最终结果，是让采样噪声更接近高斯分布，拟合的逆噪声也接近高斯分布。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('done')

def sample_imgs(ddpm, net, output_path, n_sample=80, device=device, simple_var=True):  # 加噪过程是否用简单方差
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  #  1,3,28,28
        imgs = ddpm.sample_backward(shape, net, device=device, simple_var=simple_var).detach().cpu()  # 反向去噪，生成图像
        imgs = (imgs + 1) / 2 * 255  # 将值域从[-1,1] 转化到 [0, 255]
        imgs = imgs.clamp(0, 255)  # 将tensor中所有元素的值限制在一个指定范围内。
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))
        imgs = imgs.numpy().astype(np.uint8)
        cv2.imwrite(output_path, imgs)

configs = [convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg,
           unet_1_cfg, unet_res_cfg]

if __name__=='__main__':
    os.makedirs('work_dirs', exist_ok=True)

    n_steps = 1000
    config_id = 4
    model_path = 'model_unet_res.pth'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device=device, ckpt_path=model_path)

    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, 'work_dirs/diffusion_backward.jpg', device=device)