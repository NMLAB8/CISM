import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM
import math
import torch.nn.functional as F

class FeatureConversion(nn.Module):
    """
    特征转换模块，集成到相邻的双路径层中。
    用于在频域和时域之间进行转换。
    
    参数:
        channels (int): 输入通道数
        inverse (bool): 如果为 True，使用 IFFT；否则使用 RFFT
        
    维度流:
        FFT模式 (inverse=False):
            输入: [B, C, Fr, T]
            RFFT: [B, C, Fr, T] -> [B, C, Fr, T//2 + 1]  # 复数域
            分离实虚部: [B, C, Fr, T//2 + 1] -> [B, C*2, Fr, T//2 + 1]  # 实数域
            
        IFFT模式 (inverse=True):
            输入: [B, C*2, Fr, T//2 + 1]  # 实数域
            合并实虚部: [B, C*2, Fr, T//2 + 1] -> [B, C, Fr, T//2 + 1]  # 复数域
            IRFFT: [B, C, Fr, T//2 + 1] -> [B, C, Fr, T]  # 实数域
            
        其中:
            B: 批次大小
            C: 通道数
            Fr: 频率维度
            T: 时间维度
    """    
    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse = inverse
        self.channels = channels

    def forward(self, x):
        # B: 批次大小, C: 通道数, F: 频率维度, T: 时间维度
        if self.inverse:
            # 将频域信号转换回时域
            x = x.float()
            # 分离实部和虚部
            x_r = x[:, :self.channels//2, :, :]  # 实部
            x_i = x[:, self.channels//2:, :, :]  # 虚部
            # 构建复数张量
            x = torch.complex(x_r, x_i)
            # 执行逆实数傅里叶变换
            x = torch.fft.irfft(x, dim=3, norm="ortho")
        else:
            # 将时域信号转换到频域
            x = x.float()
            # 执行实数傅里叶变换
            x = torch.fft.rfft(x, dim=3, norm="ortho")
            # 分离并拼接实部和虚部
            x_real = x.real
            x_imag = x.imag
            x = torch.cat([x_real, x_imag], dim=1)
        return x


class MultiHeadAttention(nn.Module):
    """
    多头自注意力模块
    
    参数:
        d_model (int): 输入特征维度
        num_heads (int): 注意力头数
        dropout (float): dropout率
        
    维度流:
        输入: [B, L, D]  # B: 批次大小, L: 序列长度, D: 特征维度
        1. 线性变换: [B, L, D] -> [B, L, D] (Q, K, V)
        2. 分头: [B, L, D] -> [B, H, L, D/H]  # H: 头数
        3. 注意力计算: [B, H, L, L]
        4. 加权求和: [B, H, L, D/H]
        5. 合并多头: [B, L, D]
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        参数:
            x: [B, L, D]
        返回:
            [B, L, D]
        """
        B, L, D = x.shape
        
        # 1. 线性变换并分头
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, D/H]
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, D/H]
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, D/H]
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L, L]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 3. 注意力加权求和
        out = torch.matmul(attn, V)  # [B, H, L, D/H]
        
        # 4. 合并多头
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.W_o(out)
        
        return out


class DualPathRNN(nn.Module):
    """  
    分离网络中的双路径 RNN 模块。
    实现了在频率和时间维度上的双向处理。

    参数:
        d_model (int): 输入特征维度
        expand (int): LSTM隐藏层扩展因子
        bidirectional (bool): 是否使用双向LSTM
        
    维度流:
        输入: [B, C, Fr, T]
        
        频率路径:
            1. 重排: [B, C, Fr, T] -> [B*T, Fr, C]
            2. LSTM: [B*T, Fr, C] -> [B*T, Fr, C*2]  # 双向使通道翻倍
            3. 线性层: [B*T, Fr, C*2] -> [B*T, Fr, C]
            4. 重排回: [B*T, Fr, C] -> [B, C, Fr, T]
            
        时间路径:
            1. 重排: [B, C, Fr, T] -> [B*Fr, T, C]
            2. LSTM: [B*Fr, T, C] -> [B*Fr, T, C*2]  # 双向使通道翻倍
            3. 线性层: [B*Fr, T, C*2] -> [B*Fr, T, C]
            4. 重排回: [B*Fr, T, C] -> [B, C, Fr, T]
    """
    def __init__(self, d_model, expand, bidirectional=True):
        super(DualPathRNN, self).__init__()

        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        
        # LSTM层和归一化层保持不变
        self.lstm_layers = nn.ModuleList([
            self._init_lstm_layer(self.d_model, self.hidden_size) for _ in range(2)
        ])
        self.linear_layers = nn.ModuleList([
            nn.Linear(self.hidden_size*2, self.d_model) for _ in range(2)
        ])
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(1, d_model) for _ in range(2)
        ])
        
        # 分别为频率路径和时间路径创建注意力模块
        self.freq_attention = MultiHeadAttention(d_model)  # 频率维度的注意力
        self.time_attention = MultiHeadAttention(d_model)  # 时间维度的注意力

    def _init_lstm_layer(self, d_model, hidden_size):
        """
        初始化 LSTM 层
        输入维度: d_model
        隐藏层维度: hidden_size
        输出维度: hidden_size*2 (因为bidirectional=True)
        """
        return LSTM(d_model, hidden_size, num_layers=1, bidirectional=self.bidirectional, batch_first=True)

    def forward(self, x):
        # 添加打印标志
        if not hasattr(self, 'print_once'):
            self.print_once = True
        
        # 输入x: [B, C, Fr, T]
        B, C, F, T = x.shape
        if self.print_once:
            print("\nDualPathRNN input shape:", x.shape)
        
        # 1. 频率路径处理
        original_x = x  # [B, C, Fr, T]
        x = self.norm_layers[0](x)  # [B, C, Fr, T]
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)  # [B*T, Fr, C]
        if self.print_once:
            print("Freq path after reshape:", x.shape)
        
        x, _ = self.lstm_layers[0](x)  # [B*T, Fr, C*2]
        if self.print_once:
            print("Freq path after LSTM:", x.shape)
        
        x = self.linear_layers[0](x)  # [B*T, Fr, C]
        if self.print_once:
            print("Freq path after linear:", x.shape)
        
        # 在频率维度上应用注意力 (Fr作为序列长度)
        x = self.freq_attention(x)  # [B*T, Fr, C]
        if self.print_once:
            print("Freq path after attention:", x.shape)
        
        # 恢复原始形状并添加残差连接
        x = x.view(B, T, F, C).transpose(1, 3)  # [B, C, Fr, T]
        x = x + original_x  # [B, C, Fr, T]
        if self.print_once:
            print("Freq path final output:", x.shape)

        # 2. 时间路径处理
        original_x = x  # [B, C, Fr, T]
        x = self.norm_layers[1](x)  # [B, C, Fr, T]
        x = x.transpose(1, 2).contiguous().view(B * F, C, T).transpose(1, 2)  # [B*Fr, T, C]
        if self.print_once:
            print("\nTime path after reshape:", x.shape)
        
        x, _ = self.lstm_layers[1](x)  # [B*Fr, T, C*2]
        if self.print_once:
            print("Time path after LSTM:", x.shape)
        
        x = self.linear_layers[1](x)  # [B*Fr, T, C]
        if self.print_once:
            print("Time path after linear:", x.shape)
        
        # 在时间维度上应用注意力 (T作为序列长度)
        x = self.time_attention(x)  # [B*Fr, T, C]
        if self.print_once:
            print("Time path after attention:", x.shape)
        
        # 恢复原始形状并添加残差连接
        x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)  # [B, C, Fr, T]
        x = x + original_x  # [B, C, Fr, T]
        if self.print_once:
            print("Time path final output:", x.shape)
            print("-" * 50)
            self.print_once = False

        return x  # [B, C, Fr, T]
    




class SeparationNet(nn.Module):
    """
    分离网络的主要实现。
    通过多层双路径 RNN 和特征转换来实现音频源的分离。
    
    参数:
        channels (int): 输入通道数，等于SCNet中dims[-1]
        expand (int): LSTM隐藏层扩展因子
        num_layers (int): 双路径层的数量
        
    维度流:
        输入: [B, C, Fr, T]
        
        每层处理:
        1. 双路径RNN (dp_modules):
           - 偶数层: [B, C, Fr, T] -> [B, C, Fr, T]
           - 奇数层: [B, C*2, Fr, T//2 + 1] -> [B, C*2, Fr, T//2 + 1]
           
        2. 特征转换 (feature_conversion):
           - 偶数层(FFT): [B, C, Fr, T] -> [B, C*2, Fr, T//2 + 1]
           - 奇数层(IFFT): [B, C*2, Fr, T//2 + 1] -> [B, C, Fr, T]
           
        输出: [B, C, Fr, T]
        
    注意:
        - 通道数在FFT和IFFT之间交替变化(C <-> C*2)
        - 时间维度在FFT和IFFT之间交替变化(T <-> T//2 + 1)
        - 每个双路径RNN的输入输出维度保持一致
    """
    def __init__(self, channels, expand=1, num_layers=6):
        super(SeparationNet, self).__init__()
        
        self.num_layers = num_layers

        # 创建双路径 RNN 模块列表
        # 奇数层的通道数翻倍(因为FFT后实虚部分离)
        self.dp_modules = nn.ModuleList([
            DualPathRNN(channels * (2 if i % 2 == 1 else 1), expand) 
            for i in range(num_layers)
        ])
        
        # 创建特征转换模块列表
        # 偶数层执行 FFT，奇数层执行 IFFT
        self.feature_conversion = nn.ModuleList([
            FeatureConversion(
                channels * 2, 
                inverse = False if i % 2 == 0 else True
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        """
        参数:
            x: 输入张量 [B, C, Fr, T]
               B: 批次大小
               C: 通道数
               Fr: 频率维度
               T: 时间维度
               
        返回:
            与输入相同维度的张量 [B, C, Fr, T]
        """
        # 依次通过所有双路径层和特征转换层
        for i in range(self.num_layers):
           x = self.dp_modules[i](x)  # 双路径RNN处理
           x = self.feature_conversion[i](x)  # 特征转换(FFT/IFFT)
        return x
