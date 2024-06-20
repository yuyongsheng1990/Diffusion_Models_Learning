# -*- coding: utf-8 -*-
# @Time : 2024/6/18 15:04
# @Author : yysgz
# @File : network.py
# @Project : DDPM_model_2
# @Description : U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_img_shape

# 位置编码在transformer model中用于为输入序列中的每个位置提供唯一的表示，从而保留输入序列中的位置信息。
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model:int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0  # false时，抛出异常，确保d_model是偶数

        pe = torch.zeros(max_seq_len, d_model)  # positional encoding = position * frequency encoding, 创建位置编码matrix，全零tensor
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)  # 生成位置序列
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)  # 生成频率序列
        '''
            torch.meshgrid函数生成网格，也可用于生成坐标。
            input: 两个数据类型相同的一维tensor。
            output: 两个output tensor，行数为第一个input tensor的元素个数；列数为第二个input tensor的元素个数。
                - 第一个output tensor填充的是第一个input tensor中的元素，各行元素相同。
                - 第二个output tensor填充的是第二个input tensor中的元素，各列元素相同。
                - tensor([1,1,1],[2,2,2],[3,3,3]); tensor([4,5,6],[4,5,6],[4,5,6])
        '''
        pos, two_i = torch.meshgrid(i_seq, j_seq)  # 创建网格张量，shape=(max_seq_len, d_model // 2)
        # 计算位置编码矩阵，通过对位置编码pos和频率序列进行特定变换得到
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))  # 2i表示偶数；** 幂运算；1/2 * d_model个正弦波编码
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))  # 21+1表示奇数；
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)  # torch.stack函数与torch.cat不同，拼接tensor后会插入一个新的维度，这里dim=2与dim=-1作用相同。

        self.embedding = nn.Embedding(max_seq_len, d_model)  # lookup操作：创建embedding matrix，(max_seq_len, d_model)，输入是一个索引index，输出是index对应的embedding vector。
        self.embedding.weight.data = pe  # 初始化embedding layer权重，pe shape也是(max_seq_len, d_model)
        self.embedding.requires_grad_(False)  # 阻止梯度更新

    def forward(self, t):
        return self.embedding(t)

class ResidualBlock(nn.Module):  # 残差网络块
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)  # kernel_size=3, stride=1, padding=1.
        self.bn1 = nn.BatchNorm2d(out_c)  # 适用于二维卷积网络的批归一化，x_hat = (xi-μ)/sqrt(σ^2 + ε)；x_hat * γ + β
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.activation2 = nn.ReLU()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                          nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()  # 恒等映射，将input tensor不做任何变换，直接返回

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input)  # residual connection
        x = self.activation2(x)
        return x

class ConvNet(nn.Module):  # 定义了一个CNN with residual block

    def __init__(self, n_steps, intermediate_channels=[10, 20, 40], pe_dim=10, insert_t_to_all_layers=False):
        super().__init__()
        C, H, W = get_img_shape()  # 1, 28, 28
        self.pe = PositionalEncoding(n_steps, pe_dim)  # 位置编码
        self.pe_linears = nn.ModuleList()  # 创建位置编码的线性层模块列表
        self.all_t = insert_t_to_all_layers
        if not insert_t_to_all_layers:  # 是否在所有层中插入时间t位置编码
            self.pe_linears.append(nn.Linear(pe_dim, C))

        self.residual_blocks = nn.ModuleList()  # 残差模块列表
        prev_channel = C  # previous channel
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))  # residual block
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))  # 如果插入时间编码，则t from pe_dim to prev_channel, 在输入到残差块res_block from prev_channel to channel.
            else:
                self.pe_linears.append(None)
            prev_channel = channel
        self.output_layer = nn.Conv2d(prev_channel, C, 3, 1, 1)  # 输出层

    def forward(self, x, t):
        n = t.shape[0]  # t num 时间数量
        t = self.pe(t)  # 时间位置编码 matrix
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            if m_t is not None:
                pe = m_t(t).reshape(n, -1, 1, 1)  # 与x维度保持一致
                x = x + pe
            x = m_x(x)
        x = self.output_layer(x)
        return x

class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual=False):
        super().__init__()
        self.ln = nn.LayerNorm(shape)  # 层归一化
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)  # kernel_size=3, stride=1, padding=1.
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()  # 恒等映射
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:  # 拼接在后面
            out += self.residual_conv(x)
        out = self.activation(out)
        return out

class UNet(nn.Module):  # 基于U-Net架构的CNN，包括了位置编码PE和残差连接residual connections，该网络可以用于图像生成任务。
    def __init__(self, n_steps, channels=[10, 20, 40, 80], pe_dim=10, residual=False) -> None:
        super().__init__()
        C, H, W = get_img_shape()  # 1, 28, 28
        layers = len(channels)  # 相当于 T steps=4
        Hs = [H]  # U-Net计算每一层的高和宽是为了正确构建上采样up-sampling和拼接操作，这些计算确保在解码路径中，经过上采样后的特征图与编码路径中相应层的特征图具有兼容的大小，从而可以顺利的进行拼接操作。
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1):  # 每传递一层，layer height和width就缩小一半
            cH //= 2
            cW //= 2
            Hs.append(cH)  # [28, 14, 7, 3]
            Ws.append(cW)

        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.encoders = nn.ModuleList()  # 编码器
        self.decoders = nn.ModuleList()  # 解码器
        self.pe_linears_en = nn.ModuleList()  # 位置编码 linears of 编码器
        self.pe_linears_de = nn.ModuleList()  # 位置编码 linear of decoders
        self.downs = nn.ModuleList()  # 下采样模块
        self.ups = nn.ModuleList()  # 上采样模块
        # 1. 正向编码过程
        prev_channel = C  # previous channel
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):  # 这里取不到最后一个channel，只能取到40.
            # 编码器-位置编码线性层，(pe_dim, prev_channel)
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(), nn.Linear(prev_channel, prev_channel)))
            # 编码器层
            self.encoders.append(nn.Sequential(  # 两个Unet block
                            UnetBlock((prev_channel, cH, cW),  # shape = (1, 28, 28); (10, 14, 14); (20, 7, 7)
                                               prev_channel,  # in_channel
                                               channel,       # out_channel
                                               residual=residual),
                            UnetBlock((channel, cH, cW),      # shape = (10, 28, 28); (20, 14, 14); (40, 7, 7)
                                      channel,
                                      channel,
                                      residual=residual)))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel  # 10, 20, 40

        # 中间层 mid
        self.pe_mid = nn.Linear(pe_dim, prev_channel)  # 中间层位置编码，(pe_dim, 40)
        channel = channels[-1]  # last channel = 80
        self.mid = nn.Sequential(  # 中间层模块，也是两层Unet block
            UnetBlock((prev_channel, Hs[-1], Ws[-1]),  # img图片被缩小的最小height和width。(40, 3, 3)
                      prev_channel,  # 40
                      channel,  # last channel = 80
                      residual=residual),
            UnetBlock((channel, Hs[-1], Ws[-1]),  # (80, 3, 3)
                      channel,
                      channel,
                      residual=residual),)

        # 2. 反向解码过程
        prev_channel = channel  # 80
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):  # 从-2开始倒采样，[40, 20, 10], Hs=[7, 14, 28]
            # 解码器-位置编码线性层
            self.pe_linears_de.append(nn.Linear(pe_dim,prev_channel))  # 80, 40; 20
            # 上采样层
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))  # 反卷积，kernel_size=2, stride=2, 上采样 = 增加分辨率，(80,40); (40,20); (20,10)
            # 解码器层
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2, cH, cW),  # 因为这里有拼接！！！！！layerNorm shape = (80, 7, 7); (40, 14, 14); (20, 28, 28)
                              channel * 2,            # in_channel = 80; 40; 20
                              channel,                # out_channel = 40; 20; 10
                              residual=residual),
                    UnetBlock((channel, cH, cW),      # (40, 7, 7); (20, 14, 14); (10, 28, 28)
                              channel,                # 40; 20; 10
                              channel,                # 40; 20; 10
                              residual=residual)))
            prev_channel = channel  # 40; 20; 10
        # 输出层
        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)  # (10, 1), kernel_size=3, stride=1, padding=1

    def forward(self, x, t):  # t_tensor = [t]*n
        n = t.shape[0]  # t num 时间数量
        t = self.pe(t)  # 时间位置编码 matrix
        encoder_outs = []  # 编码器输出
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders, self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)  # 与x维度保持一致
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)
        pe = self.pe_mid(t).reshape(n, -1, 1, 1)  # 中间层，channel from 40 to 80
        x = self.mid(x + pe)  # (80,3,3)
        # 解码过程
        for pe_linear, decoder, up, encoder_out, in zip(self.pe_linears_de, self.decoders, self.ups, encoder_outs[::-1]):  # 倒序读取编码器输出, 3个 (40,7,7); (20,14,14); (10,28,28)
            pe = pe_linear(t).reshape(n, -1, 1, 1)  # 位置编码
            x = up(x)  # 上采样，(80,3,3) -> (40,3,3); (40,7,7) -> (20,7,7); (20,14,14) -> (10,14,14)
            pad_x = encoder_out.shape[2] - x.shape[2]  # 计算第三个维度需要填充的num：通常是高度H，(40,7,7) - (40,3,3); (20,14,14) - (20,7,7); (10,28,28) - (10,14,14)
            pad_y = encoder_out.shape[3] - x.shape[3]  # 计算第四个维度需要填充的num：通常是宽度，
            # F.pad对x进行填充操作，后面四元组参数是四个边界的填充量
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2))  # (40,3,3)-> (40,7,7); (20,7,7)-> (20,14,14); (10,14,14) -> (10,28,28)
            x = torch.cat((encoder_out, x), dim=1)  # dim=1 拼接 40->80; 20 -> 40; 10 -> 20
            x = decoder(x + pe)  # (80,7,7) -> (40,7,7); (40,14,14) -> (20,14,14); (20,28,28) -> (10,28,28)
        x = self.conv_out(x)  # 10 -》 1
        return x

# 定义一个配置字典，主要是为了方便管理和传递模型的超参数和配置。这种配置字典可以用于初始化模型的各个参数。
convnet_small_cfg = {  # configuration
    'type' : 'ConvNet',
    'intermediate_channels' : [10, 20],
    'pe_dim' : 128
}

convnet_medium_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}

convnet_big_cfg = {
    'type': 'ConvNet',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160,160],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}

unet_1_cfg = {'type': 'UNet',
              'channels': [10, 20, 40, 80],
              'pe_dim': 128}p

unet_res_cfg = {'type': 'UNet',
                'channels': [10, 20, 40, 80],
                'pe_dim': 128,
                'residual': True}

def build_network(config: dict, n_steps):
    network_type = config.pop('type')
    if network_type == 'ConvNet':
        network_cls = ConvNet
    elif network_type == 'UNet':
        network_cls = UNet

    network = network_cls(n_steps, **config)
    return network

# For examples:
# if __name__=='__main__':
#     model = build_network(convnet_medium_cfg, 10)
