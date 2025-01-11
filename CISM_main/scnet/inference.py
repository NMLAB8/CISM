import os
import time
import numpy as np
import torch
import soundfile as sf
from .SCNet import SCNet
from .utils import load_model, convert_audio
from .apply import apply_model
from ml_collections import ConfigDict
import argparse
import yaml



class Seperator:
    """
    音频分离器类，用于处理音频文件的源分离。
    
    属性:
        separator: 加载的 SCNet 模型
        device: 用于推理的设备（GPU 或 CPU）
        instruments: 要分离的音频源列表
    """
    def __init__(self, model, checkpoint_path):
        """
        初始化分离器。
        
        参数:
            model: SCNet 模型实例
            checkpoint_path: 模型检查点文件的路径
        """
        # 加载预训练模型
        self.separator = load_model(model, checkpoint_path)

        # 设置设备（优先使用 GPU）
        if torch.cuda.device_count():
            self.device = torch.device('cuda')
        else:
            print("WARNING, using CPU")
            self.device = torch.device('cpu')
        self.separator.to(self.device)

    @property
    def instruments(self):
        """返回要分离的音频源列表"""
        return ['other', 'solo']

    def raise_aicrowd_error(self, msg):
        """抛出自定义错误"""
        raise NameError(msg)

    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        对单个音频文件进行源分离。

        参数:
            mixed_sound_array: 混合音频数据数组
            sample_rate: 音频采样率

        返回:
            separated_music_arrays: 包含分离后各个源的字典
            output_sample_rates: 包含各个源采样率的字典
        """
        # 将输入音频转换为 PyTorch 张量
        mix = torch.from_numpy(np.asarray(mixed_sound_array.T, np.float32))

        # 将音频数据移动到指定设备
        mix = mix.to(self.device)
        mix_channels = mix.shape[0]
        # 转换音频采样率和通道数
        mix = convert_audio(mix, sample_rate, 44100, self.separator.audio_channels)

        # 记录开始时间
        b = time.time()
        # 计算音频的均值和标准差用于归一化
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std

        # 使用模型进行分离
        with torch.no_grad():
            estimates = apply_model(self.separator, mix[None], overlap=0.5, progress=False)[0]

        # 打印处理时间和其他信息
        print(time.time() - b, mono.shape[-1] / sample_rate, mix.std(), estimates.std())

        # 反归一化
        estimates = estimates * std + mean

        # 将音频转换回原始采样率和通道数
        estimates = convert_audio(estimates, 44100, sample_rate, mix_channels)

        # 准备返回结果
        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            idx = self.separator.sources.index(instrument)
            separated_music_arrays[instrument] = torch.squeeze(estimates[idx]).detach().cpu().numpy().T
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates

    def load_audio(self, file_path):
        """
        加载音频文件。

        参数:
            file_path: 音频文件路径

        返回:
            data: 音频数据
            sample_rate: 采样率
        """
        try:
            data, sample_rate = sf.read(file_path, dtype='float32')
            return data, sample_rate
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            raise

    def save_sources(self, sources, output_sample_rates, save_dir):
        """
        保存分离后的音频源。

        参数:
            sources: 分离后的音频源字典
            output_sample_rates: 输出采样率字典
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        for name, src in sources.items():
            save_path = os.path.join(save_dir, f'{name}.wav')
            sf.write(save_path, src, output_sample_rates[name])
            print(f"Saved {name} to {save_path}")

    def process_directory(self, input_dir, output_dir):
        """
        处理整个目录中的音频文件。

        参数:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
        """
        for entry in os.listdir(input_dir):
            entry_path = os.path.join(input_dir, entry)
            # 处理目录中的 mixture.wav 文件
            if os.path.isdir(entry_path):
                mixture_path = os.path.join(entry_path, 'mixture.wav')
                if os.path.isfile(mixture_path):
                    print(f"Processing {mixture_path}")
                    entry_name = os.path.basename(entry)
                else:
                    continue
            # 处理单个 WAV 文件
            elif os.path.isfile(entry_path) and entry_path.lower().endswith('.wav'):
                print(f"Processing {entry_path}")
                mixture_path = entry_path
                entry_name = os.path.splitext(os.path.basename(entry))[0]
            else:
                continue

            # 加载并处理音频文件
            mixed_sound_array, sample_rate = self.load_audio(mixture_path)
            separated_music_arrays, output_sample_rates = self.separate_music_file(mixed_sound_array, sample_rate)
            save_dir = os.path.join(output_dir, entry_name)
            self.save_sources(separated_music_arrays, output_sample_rates, save_dir)

def parse_args():
    """
    解析命令行参数。

    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="Music Source Separation using SCNet")
    parser.add_argument('--input_dir', type=str, help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, help='Output directory to save separated sources')
    parser.add_argument('--config_path', type=str, default='./conf/config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint_path', type=str, default='./result/checkpoint.th', help='Path to model checkpoint file')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 加载配置文件
    with open(args.config_path, 'r') as file:
          config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # 初始化模型
    model = SCNet(**config.model)
    model.eval()
    # 创建分离器实例并处理音频文件
    seperator = Seperator(model, args.checkpoint_path)
    seperator.process_directory(args.input_dir, args.output_dir)
