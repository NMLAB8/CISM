import os
import time
import torch
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import yaml
from ml_collections import ConfigDict
import argparse
from tqdm import tqdm
from .SCNet import SCNet
from .inference import Seperator
from .log import logger
import resampy

class Evaluator:
    """用于评估音频源分离模型性能的评估器类"""
    
    def __init__(self, model, checkpoint_path, config):
        self.separator = Seperator(model, checkpoint_path)
        # 加载检查点以获取 epoch 信息
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.epoch = self.checkpoint.get('epoch', 'unknown')  # 如果没有 epoch 信息则返回 'unknown'
        # 保存频带分割比例信息
        self.band_sr = config.model.band_SR
    
    def calculate_si_sdr(self, reference, estimated):
        """计算 Scale-Invariant SDR"""
        # 确保输入是一维数组
        if reference.ndim > 1:
            reference = np.mean(reference, axis=1)
        if estimated.ndim > 1:
            estimated = np.mean(estimated, axis=1)
        
        reference = reference - np.mean(reference)
        estimated = estimated - np.mean(estimated)
        
        # 计算 scaling factor
        alpha = np.dot(reference, estimated) / np.dot(reference, reference)
        
        # 计算 SI-SDR
        scaled = alpha * reference
        e_true = scaled
        e_res = estimated - scaled
        
        return 10 * np.log10(np.sum(e_true ** 2) / (np.sum(e_res ** 2) + 1e-8))

    def calculate_stoi(self, reference, estimated, sr):
        """计算 STOI (Short-Time Objective Intelligibility)"""
        # STOI 需要单声道信号
        if reference.ndim > 1:
            reference = np.mean(reference, axis=1)
        if estimated.ndim > 1:
            estimated = np.mean(estimated, axis=1)
            
        # STOI 需要 10000Hz 以上的采样率
        if sr < 10000:
            reference = resampy.resample(reference, sr, 10000)
            estimated = resampy.resample(estimated, sr, 10000)
            sr = 10000
            
        return stoi(reference, estimated, sr, extended=False)

    def load_audio_pair(self, mixture_path, source_paths):
        """
        加载混合音频和对应的源音频
        
        参数:
            mixture_path: 混合音频文件路径
            source_paths: 源音频文件路径字典
            
        返回:
            mixture_audio: 混合音数据
            source_audios: 源音频数据字典
            sample_rate: 采样率
        """
        mixture_audio, rate = sf.read(mixture_path)
        source_audios = {}
        for name, path in source_paths.items():
            audio, sr = sf.read(path)
            assert sr == rate, f"采样率不匹配: {sr} != {rate}"
            source_audios[name] = audio
        return mixture_audio, source_audios, rate

    def calculate_pesq(self, reference, estimated, sr):
        """
        计算 PESQ 分数
        首先将音频重采样到 16000Hz，因为 PESQ 只支持 8000Hz 或 16000Hz
        """
        target_sr = 16000  # 目标采样率

        if reference.ndim > 1:
            reference = np.mean(reference, axis=1)
        if estimated.ndim > 1:
            estimated = np.mean(estimated, axis=1)

        # 重采样到 16000Hz
        if sr != target_sr:
            reference = resampy.resample(reference, sr, target_sr)
            estimated = resampy.resample(estimated, sr, target_sr)
        
        return pesq(target_sr, reference, estimated, 'wb')

    def calculate_mse(self, reference, estimated):
        """计算均方误差 (MSE)"""
        # 确保输入是一维数组
        if reference.ndim > 1:
            reference = np.mean(reference, axis=1)
        if estimated.ndim > 1:
            estimated = np.mean(estimated, axis=1)
        
        return np.mean((reference - estimated) ** 2)

    def calculate_ldr(self, reference, estimated):
        """计算对数密度比 (LDR)"""
        # 确保输入是一维数组
        if reference.ndim > 1:
            reference = np.mean(reference, axis=1)
        if estimated.ndim > 1:
            estimated = np.mean(estimated, axis=1)
        
        return np.mean(np.abs(np.log(np.abs(reference) + 1e-8) - 
                             np.log(np.abs(estimated) + 1e-8)))

    def calculate_metrics(self, references, estimates, sample_rate, rtf):
        """计算所有评估指标"""
        metrics = {}
        
        # 为每个源计算指标
        pesq_scores = []
        stoi_scores = []
        si_sdr_scores = []
        mse_scores = []
        ldr_scores = []
        
        for ref, est in zip(references, estimates):
            # 确保长度匹配
            min_len = min(len(ref), len(est))
            ref = ref[:min_len]
            est = est[:min_len]
            
            # MSE
            mse = self.calculate_mse(ref, est)
            mse_scores.append(mse)
            
            # LDR
            ldr = self.calculate_ldr(ref, est)
            ldr_scores.append(ldr)
            
            # SI-SDR
            si_sdr = self.calculate_si_sdr(ref, est)
            si_sdr_scores.append(si_sdr)
            
            # STOI
            try:
                stoi_score = self.calculate_stoi(ref, est, sample_rate)
                stoi_scores.append(stoi_score)
            except Exception as e:
                logger.warning(f"计算STOI时出错: {e}")
            
            # PESQ
            try:
                pesq_score = self.calculate_pesq(ref, est, sample_rate)
                pesq_scores.append(pesq_score)
            except Exception as e:
                logger.warning(f"计算PESQ时出错: {e}")
        
        # 计算平均值
        metrics['PESQ'] = np.mean(pesq_scores) if pesq_scores else 0.0
        metrics['STOI'] = np.mean(stoi_scores) if stoi_scores else 0.0
        metrics['SI-SDR'] = np.mean(si_sdr_scores)
        metrics['MSE'] = np.mean(mse_scores)
        metrics['LDR'] = np.mean(ldr_scores)
        metrics['RTF'] = rtf
        
        return metrics

    def evaluate_folder(self, test_dir):
        """评估测试文件夹中的所有样本"""
        all_metrics = {
            'PESQ': [], 'STOI': [], 'SI-SDR': [], 
            'LDR': [], 'MSE': [], 'RTF': []
        }
        
        detailed_results = []
        
        # 遍历测试文件夹中的所有样本
        for sample_dir in tqdm(os.listdir(test_dir), desc="Processing samples"):
            sample_path = os.path.join(test_dir, sample_dir)
            if not os.path.isdir(sample_path):
                continue
                
            mixture_path = os.path.join(sample_path, 'mixture.wav')
            source_paths = {
                'solo': os.path.join(sample_path, 'solo.wav'),
                'other': os.path.join(sample_path, 'other.wav')
            }
            
            try:
                # 加载音频文件
                mixture_audio, source_audios, sample_rate = self.load_audio_pair(
                    mixture_path, source_paths)
                
                # 检查音频长度，如果太长则截断到30秒
                max_duration = 30  # 最大处理时长改为30秒
                max_samples = max_duration * sample_rate
                
                if len(mixture_audio) > max_samples:
                    logger.warning(f"音频 {sample_dir} 太长 ({len(mixture_audio)/sample_rate:.2f}s)，将截断至 {max_duration}s")
                    mixture_audio = mixture_audio[:max_samples]
                    source_audios = {k: v[:max_samples] for k, v in source_audios.items()}
                
                # 记录开始时间
                start_time = time.time()
                
                # 使用模型进行分离
                estimates, _ = self.separator.separate_music_file(
                    mixture_audio, sample_rate)
                
                # 计算处理时间和 RTF
                process_time = time.time() - start_time
                audio_length = len(mixture_audio) / sample_rate
                rtf = process_time / audio_length
                
                # 将源和估计转换为评估所需的格式
                references = np.stack([source_audios['solo'], source_audios['other']])
                estimated = np.stack([estimates['solo'], estimates['other']])
                
                # 计算所有指标
                metrics = self.calculate_metrics(references, estimated, sample_rate, rtf)
                
                # 保存指标
                for key in metrics:
                    all_metrics[key].append(metrics[key])
                
                # 保存当前样本的详细结果
                sample_result = {
                    'sample_name': sample_dir,
                    'duration': audio_length,  # 添加音频时长信息
                    'PESQ': metrics['PESQ'],
                    'STOI': metrics['STOI'],
                    'SI-SDR': metrics['SI-SDR'],
                    'LDR': metrics['LDR'],
                    'MSE': metrics['MSE'],
                    'RTF': metrics['RTF']
                }
                detailed_results.append(sample_result)
                
                # 打印当前样本的指标
                logger.info(f"\nResults for sample {sample_dir} (duration: {audio_length:.2f}s):")
                logger.info(f"PESQ: {metrics['PESQ']:.4f}")
                logger.info(f"STOI: {metrics['STOI']:.4f}")
                logger.info(f"SI-SDR: {metrics['SI-SDR']:.4f}")
                logger.info(f"LDR: {metrics['LDR']:.4f}")
                logger.info(f"MSE: {metrics['MSE']:.4f}")
                logger.info(f"RTF: {metrics['RTF']:.4f}")
                
            except Exception as e:
                logger.error(f"处理样本 {sample_dir} 时出错: {e}")
                continue
        
        # 计算平均值
        final_metrics = {k: np.mean(v) for k, v in all_metrics.items() if len(v) > 0}
        
        return final_metrics, detailed_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate SCNet model")
    parser.add_argument('--test_dir', type=str, required=True,
                      help='Directory containing test data')
    parser.add_argument('--config_path', type=str, 
                      default='./conf/config.yaml',
                      help='Path to config file')
    parser.add_argument('--checkpoint_path', type=str, 
                      default='./result/checkpoint.th',
                      help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str,
                      default='./evaluation_results',
                      help='Directory to save evaluation results')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载配置
    with open(args.config_path, 'r') as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    # 初始化模型
    model = SCNet(**config.model)
    model.eval()

    # 创建评估器
    evaluator = Evaluator(model, args.checkpoint_path, config)
    
    # 运行评估
    metrics, detailed_results = evaluator.evaluate_folder(args.test_dir)
    
    # 生成时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    results_file = os.path.join(args.output_dir, f'evaluation_results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        # 写入总体评估结果，包含 epoch 和频带分割信息
        f.write(f"=== Overall Evaluation Results (Epoch {evaluator.epoch}, Band SR {evaluator.band_sr}) ===\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        # 写入每个样本的详细结果
        f.write("\n=== Detailed Results for Each Sample ===\n")
        for result in detailed_results:
            f.write(f"\nSample: {result['sample_name']}\n")
            f.write(f"PESQ: {result['PESQ']:.4f}\n")
            f.write(f"STOI: {result['STOI']:.4f}\n")
            f.write(f"SI-SDR: {result['SI-SDR']:.4f}\n")
            f.write(f"LDR: {result['LDR']:.4f}\n")
            f.write(f"MSE: {result['MSE']:.4f}\n")
            f.write(f"RTF: {result['RTF']:.4f}\n")
    
    # 打印结果
    logger.info(f"\nEvaluation Results saved to: {results_file}")
    logger.info("\nOverall Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
