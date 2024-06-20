# -*- coding: utf-8 -*-
# @Time : 2024/6/18 9:33
# @Author : yysgz
# @File : ddpm.py
# @Project : DDPM_model_2
# @Description :

import torch

class DDPM():

    def __init__(self, device, n_steps: int, min_beta: float=0.0001, max_beta: float=0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # linspace用于创建等间隔数值序列的函数；加噪公式：β逐渐变大
        alphas = 1 - betas  # α逐渐变小，最终xT几乎为0，picture满足均值为0，方差为I的标准正态分布。
        alpha_bars = torch.empty_like(alphas)  # αi连乘得到alpha_bars，这里用数组存储每次乘积结果
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps  #  T steps加噪和去噪
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        alpha_prev = torch.empty_like(alpha_bars)  # previous: a_{t-1}_bars
        alpha_prev[1:] = alpha_bars[0: n_steps - 1]
        alpha_prev[0] = 1  # 认为定义alpha_0 = 1, 这样去噪估计均值μ时 1-αlpha_t=0，就不影响结果了
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)  # 这是估计去噪过程的均值μ_tilde = coef1 * xt + coef2 * x0.

    # 正向扩散过程：加噪公式，xt = sqrt(alpha_t_bar) * x0 + sqrt(1-alpha_t_bar) * epsilon_t
    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)  # get t时刻 alpha_t_bar
        if eps is None:
            eps = torch.randn_like(x)  # 噪声 noise epsilon
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res  # xt，加噪后达到正态分布的噪声图像

    # 反向生成过程：去噪。
    def sample_backward(self, img_shape, net, device, simple_var=True, clip_x0=True):
        x = torch.randn(img_shape).to(device)  # 任意一个从标准正态分布中采样出来shape形状的纯噪声图像xT。
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1): # 反向过程，从xt -> x0
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)  # 每一步去噪
        return x  # 生成的图像 x0_tilde

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):
        n = x_t.shape[0]  # picture num
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)  # t应该时一个value，格式转换成*n的数组
        eps = net(x_t, t_tensor)  # model θ 估计的随机噪声 noise epsilon，这是DDPM反向过程中唯一不知道的变量，最后loss也是围绕着它进行优化！

        if t == 0:
            noise = 0
        else:  # t > 1
            if simple_var:  # 控制方差项取值方式
                var = self.betas[t]  # 这是加噪过程的方差呀！
            else:
                # 去噪过程方差：β_tilde = (1-alpha_{t-1}_bar) / (1-alpha_t_bar) * beta_t
                var = (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]) * self.betas[t]  # 这些变量值都是知道的
            # 获取方差后，我们在随机采样一个噪声noise
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)  # 随机采样noise * 标准差

        if clip_x0:
            # 由加噪公式反推出 x0 = (x_t - sqrt(1-alpha_t_bar) * eps_t) / sqrt(alpha_t_bar)
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)  # torch.clip函数用于将tensor value限制在指定范围内
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0  # 计算去噪过程均值μ_tilde
        else:
            # 将加噪过程x0带入上式，得到的xt = (sqrt(alpha_t))^{-1} * (x_t - (1-alpha_t)/sqrt(1-alpha_t_bar) * eps_t)
            mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t

def visualize_forward():
    import cv2  # pip install opencv-python
    import einops  # 用于tensor操作的package，包括reshape、broadcast、split、merge等
    import numpy as np
    from dataset import get_downloader

    # 设置超参数
    n_steps = 100
    batch_size = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = get_downloader(batch_size)  # batch_size
    x, _ = next(iter(dataloader))  # iter将dataloader转换为一个迭代器；next用于获取下一个batch

    ddpm = DDPM(device, n_steps)
    x_ts = []  # 存储加噪图像x_t
    percents = torch.linspace(0, 0.99, 10)  # 选取x_t时刻的概率：[0, 0.11, 0.22, 0.33,.., 0.99]
    for percent in percents:
        t = torch.tensor([int(n_steps * percent)])  # 0, 11, 22, 33,..., 99
        t = t.unsqueeze(1)
        x_t = ddpm.sample_forward(x, t)  # 加噪图像xt
        x_ts.append(x_t)
    res = torch.stack(x_ts, 0)  # 噪声图像 x_t list堆叠成
    res = einops.rearrange(res, 'n1 n2 c h w -> (n2 h) (n1 w) c')  # rearrange对tensor res进行重新排列，c-channel, h-height, w-width -> new dimension: n2*h, n1*w, c
    res = (res.clip(-1,1) + 1) / 2 * 255  # 值域被映射到[0, 255]
    res = res.cpu().numpy().astype(np.uint8)

    cv2.imwrite('work_dirs/diffusion_forward.jpg', res)


def main():
    visualize_forward()  # 可视化前向过程

if __name__ == "__main__":
    main()
