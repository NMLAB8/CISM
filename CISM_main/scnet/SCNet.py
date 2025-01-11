import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .separation import SeparationNet
import typing as tp
import math

# Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

# 频率注意力模块，用于增强不同频率成分的重要特征。
class FrequencyAttention(nn.Module):
    """
    频率注意力模块，用于增强不同频率成分的重要特征。
    
    参数:
        channels (int): 输入通道数
        reduction (int): 注意力计算时的通道压缩比例
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 在时间维度上平均池化
        
        # 两层MLP，用于学习频率注意力权重
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        参数:
            x: 输入张量，维度为 [B*F, C, T] 或 [B, C, F, T]
        返回:
            加权后的特征图，维度与输入相同
        """
        if not hasattr(self, 'print_once'):
            self.print_once = True  # 初始化打印标志
            
        if x.dim() == 3:
            b_f, c, t = x.size()
            if self.print_once:
                print(f"FreqAttn 3D input shape: [{b_f}, {c}, {t}]")
                print(f"FreqAttn after pool shape: {self.avg_pool(x).shape}")
                print(f"FreqAttn after squeeze shape: {self.avg_pool(x).squeeze(-1).shape}")
                print(f"FreqAttn after fc shape: {self.fc(self.avg_pool(x).squeeze(-1)).shape}")
                print(f"FreqAttn final attention shape: {self.fc(self.avg_pool(x).squeeze(-1)).unsqueeze(-1).shape}")
                print(f"FreqAttn output shape: {x.shape}")
                self.print_once = False
            
            y = self.avg_pool(x)
            y = y.squeeze(-1)
            y = self.fc(y)
            y = y.unsqueeze(-1)
            return x * y.expand_as(x)
            
        elif x.dim() == 4:
            b, c, f, t = x.size()
            if self.print_once:
                print(f"FreqAttn 4D input shape: [{b}, {c}, {f}, {t}]")
                print(f"FreqAttn after mean shape: {torch.mean(x, dim=-1).shape}")
                print(f"FreqAttn after transpose1 shape: {torch.mean(x, dim=-1).transpose(1, 2).shape}")
                print(f"FreqAttn after fc&transpose2 shape: {self.fc(torch.mean(x, dim=-1).transpose(1, 2)).shape}")
                print(f"FreqAttn final attention shape: {self.fc(torch.mean(x, dim=-1).transpose(1, 2)).unsqueeze(-1).shape}")
                print(f"FreqAttn output shape: {x.shape}")
                self.print_once = False
            
            y = torch.mean(x, dim=-1)
            y = y.transpose(1, 2)
            y = self.fc(y)
            y = y.transpose(1, 2)
            y = y.unsqueeze(-1)
            return x * y.expand_as(x)
        
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

# 卷积模块，用于 SD 块中
class ConvolutionModule(nn.Module):
    """
    SD 块中的卷积模块。
    
    参数:    
        channels (int): 输入/输出通道数。
        depth (int): 残差分支中的层数。
        compress (float): 通道压缩量。
        kernel (int): 卷积核大小。
    """
    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        norm = lambda d: nn.GroupNorm(1, d)
        
        # 添加频率注意力模块
        self.freq_attention = FrequencyAttention(channels)
        
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            padding = (kernel // 2)
            mods = [
                norm(channels),
                nn.Conv1d(channels, hidden_size*2, kernel, padding=padding),
                nn.GLU(1),
                nn.Conv1d(hidden_size, hidden_size, kernel, padding=padding, groups=hidden_size),
                norm(hidden_size),
                Swish(),
                nn.Conv1d(hidden_size, channels, 1),
            ]
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        if not hasattr(self, 'print_once'):
            self.print_once = True
            
        if self.print_once:
            print(f"\nConvModule input shape: {x.shape}")
            residual = self.layers[0](x)
            print(f"ConvModule after layer 0 shape: {residual.shape}")
            print(f"ConvModule after attention 0 shape: {self.freq_attention(residual).shape}")
            print(f"ConvModule output shape: {x.shape}\n")
            self.print_once = False
            
        for layer in self.layers:
            residual = layer(x)
            residual = self.freq_attention(residual)
            x = x + residual
        return x

# 融合层，用于解码器中
class FusionLayer(nn.Module):
    """
    解码器中的融合层。

    参数:
    - channels (int): 输入通道数。
    - kernel_size (int, optional): 卷积层的核大小，默认为 3。
    - stride (int, optional): 卷积层的步幅，默认为 1。
    - padding (int, optional): 卷积层的填充，默认为 1。
    """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(FusionLayer, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)

    def forward(self, x, skip=None):
        if skip is not None:
            x += skip
        x = x.repeat(1, 2, 1, 1)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x

# 稀疏下采样层，用于处理不同频率带
class SDlayer(nn.Module):
    """
    实现稀疏下采样层，用于分别处理不同频率带。

    参数:
    - channels_in (int): 输入通道数。
    - channels_out (int): 输出通道数。
    - band_configs (dict): 每个频率带的配置字典。
    """
    def __init__(self, channels_in, channels_out, band_configs):
        super(SDlayer, self).__init__()

        # 为每个频率带初始化卷积层
        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            self.convs.append(nn.Conv2d(channels_in, channels_out, (config['kernel'], 1), (config['stride'], 1), (0, 0)))
            self.strides.append(config['stride'])
            self.kernels.append(config['kernel'])
        
        # 保存采样率比例以确定分割点
        self.SR_low = band_configs['low']['SR']
        self.SR_mid = band_configs['mid']['SR']

    def forward(self, x):
        B, C, Fr, T = x.shape
        # 根据采样率定义分割点
        splits = [
            (0, math.ceil(Fr * self.SR_low)),
            (math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))), 
            (math.ceil(Fr * (self.SR_low + self.SR_mid)), Fr)
        ]

        # 使用相应的卷积处理每个频率带
        outputs = []
        original_lengths = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end-start)
            current_length = extracted.shape[2]

            # 填充
            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            padded = F.pad(extracted, (0, 0, pad_left, pad_right))

            output = conv(padded)
            outputs.append(output)

        return outputs, original_lengths

# 稀疏上采样层，用于解码器中
class SUlayer(nn.Module):
    """
    解码器中的稀疏上采样层。

    参数:
    - channels_in: 输入通道数。
    - channels_out: 输出通道数。
    - convtr_configs: 包含转置卷积配置的字典。
    """
    def __init__(self, channels_in, channels_out, band_configs):
        super(SUlayer, self).__init__()

        # 为每个频率带初始化卷积层
        self.convtrs = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, channels_out, [config['kernel'], 1], [config['stride'], 1])
            for _, config in band_configs.items()
        ])

    def forward(self, x, lengths, origin_lengths):
        B, C, Fr, T = x.shape
        # 根据输入长度定义分割点
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None)
        ]
        # 使用相应的卷积处理每个频率带
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            # 计算距离以对输出进行对称修剪到原始长度
            current_Fr_length = out.shape[2] 
            dist = abs(origin_lengths[idx] - current_Fr_length) // 2

            # 对出进行对称修剪到原始长度
            trimmed_out = out[:, :, dist:dist + origin_lengths[idx], :]

            outputs.append(trimmed_out)

        # 沿频率维度连接修剪后的输出以返回最终张量
        x = torch.cat(outputs, dim=2)
 
        return x

# SD 块，在编码器中实现
class SDblock(nn.Module):
    """
    在编码器中实现的简化稀疏下采样块。
    
    参数:
    - channels_in (int): 输入通道数。
    - channels_out (int): 输出通道数。
    - band_config (dict): SDlayer 的配置，指定频带分割和卷积。
    - conv_config (dict): 应用于每个频带的卷积模块配置。
    - depths (list of int): 指定低、中、高频带卷积深度的列表。
    """
    def __init__(self, channels_in, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3):
        super(SDblock, self).__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)
        
        # 根据深度动态创建每个频带的卷积模块
        self.conv_modules = nn.ModuleList([
            ConvolutionModule(channels_out, depth, **conv_config) for depth in depths
        ])
        # 设置卷积核大小为奇数
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)

    def forward(self, x):
        if not hasattr(self, 'print_once'):
            self.print_once = True
            
        # 获取基本数据
        bands, original_lengths = self.SDlayer(x)
        
        if self.print_once:
            print(f"\nSDblock input shape: {x.shape}")
            print("Band shapes after SDlayer:")
            for i, band in enumerate(bands):
                print(f"Band {i} shape: {band.shape}")
        
        # 处理每个频带
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)
        ]
        
        if self.print_once:
            print("\nBand shapes after conv_modules:")
            for i, band in enumerate(bands):
                print(f"Band {i} shape: {band.shape}")
        
        # 获取长度并连接频带
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        
        if self.print_once:
            print(f"Full band shape after cat: {full_band.shape}")
        
        # 保存skip connection并应用全局卷积
        skip = full_band
        output = self.globalconv(full_band)
        
        if self.print_once:
            print(f"SDblock output shape: {output.shape}")
            print(f"Skip connection shape: {skip.shape}")
            print(f"Lengths: {lengths}")
            print(f"Original lengths: {original_lengths}\n")
            self.print_once = False
            
        return output, skip, lengths, original_lengths

# SCNet 主体，实现了整个 SCNet 模型
class SCNet(nn.Module):
    """
    SCNet 的实现：用于音乐源分离的稀疏压缩网络。论文: https://arxiv.org/abs/2401.13276.pdf

    参数:
    - sources (List[str]): 要分离的源列表。
    - audio_channels (int): 音频通道数。
    - nfft (int): 用于确定输入频率维度的 FFT 数量。
    - hop_size (int): STFT 的跳跃大小。
    - win_size (int): STFT 的窗口大小。
    - normalized (bool): 是否对 STFT 进行归一化。
    - dims (List[int]): 每个块的通道维度列表。
    - band_SR (List[float]): 每个频带的比例。
    - band_stride (List[int]): 每个频带的下采样比率。
    - band_kernel (List[int]): 每个频带下采样卷积的核大小。
    - conv_depths (List[int]): 指定每个 SD 块中卷积模块数量的列表。
    - compress (int): 卷积模块的压缩因子。
    - conv_kernel (int): 卷积模块中卷积层的核大小。
    - num_dplayer (int): 双路径 RNN 的层数。
    - expand (int): 双路径 RNN 中的扩展因子，默认为 1。
    """
    def __init__(self,
                 sources = ['other', 'solo'],
                 audio_channels = 2,
                 # 主结构
                 dims = [4, 32, 64, 128], # SCNet-large 中 dims = [4, 64, 128, 256]
                 # STFT
                 nfft = 4096,
                 hop_size = 1024,
                 win_size = 4096,
                 normalized = True,
                 # SD/SU 层
                 band_SR = [0.175, 0.392, 0.433],     #分割比例----------------------------
                 band_stride = [1, 4, 16],             
                 band_kernel = [3, 4, 16],               
                 # 卷积模块
                 conv_depths = [3,2,1], 
                 compress = 4, 
                 conv_kernel = 3,
                 # 双路径 RNN
                 num_dplayer = 6,
                 expand = 1,
                ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }
    
        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': normalized
        }

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for index in range(len(dims)-1):
            enc = SDblock(
                    channels_in = dims[index], 
                    channels_out = dims[index+1], 
                    band_configs = self.band_configs,
                    conv_config = self.conv_config,
                    depths = conv_depths
                    )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayer(channels = dims[index+1]),
                SUlayer(
                    channels_in = dims[index+1],
                    channels_out = dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs = self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNet(
            channels = dims[-1],
            expand = expand,
            num_layers = num_dplayer,
        )        

    def forward(self, x):
        # B, C, L = x.shape
        B = x.shape[0]
        # 在初始填充中，确保 STFT 后的帧数（T 维度的长度）为偶数，
        # 以便在分离网络中使用 RFFT 操作。
        padding = self.hop_length - x.shape[-1] % self.hop_length
        if (x.shape[-1] + padding) // self.hop_length % 2 == 0:
            padding += self.hop_length
        x = F.pad(x, (0, padding))
  
        # STFT
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self.stft_config, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0]//self.audio_channels, x.shape[3]*self.audio_channels, x.shape[1], x.shape[2])
    
        B, C, Fr, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        
        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # 编码器
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # 分离
        x = self.separation_net(x)

        # 解码器
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        # 输出
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T) 
        x = x * std[:, None] + mean[:, None]
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        x = torch.istft(x, **self.stft_config)
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)
    
        x = x[:, :, :, :-padding]
        
        return x
