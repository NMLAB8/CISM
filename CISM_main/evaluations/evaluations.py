import os
import numpy as np
import torch
import soundfile as sf
import time
from pesq import pesq
import museval
from scipy import signal
import yaml
from ml_collections import ConfigDict
from typing import Dict, Tuple, List
import sys
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scnet.SCNet import SCNet
from scnet.inference import Seperator

class ModelEvaluator:
    """用于评估音源分离模型性能的评估器类"""
    
    def __init__(self, model_path: str, config_path: str):
        """
        初始化评估器
        
        参数:
            model_path: 模型检查点路径
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            
        # 初始化模型
        self.model = SCNet(**self.config.model)
        self.separator = Seperator(self.model, model_path)
        
    def calculate_sdr_sir_sar_isr(self, reference: np.ndarray, estimated: np.ndarray) -> Tuple[float, float, float, float]:
        """
        计算SDR、SIR、SAR和ISR指标
        
        参数:
            reference: 参考音频信号 [samples, channels]
            estimated: 估计的音频信号 [samples, channels]
            
        返回:
            sdr, sir, sar, isr: 计算得到的四个指标值
        """
        # 检查信号是否为全零
        if np.all(np.abs(reference) < 1e-10):
            print("警告：参考信号接近零")
            return 0.0, 0.0, 0.0, 0.0
        
        if np.all(np.abs(estimated) < 1e-10):
            print("警告：估计信号接近零")
            return 0.0, 0.0, 0.0, 0.0
        
        # 添加小的噪声以避免数值问题
        eps = 1e-10
        reference = reference + eps * np.random.randn(*reference.shape)
        estimated = estimated + eps * np.random.randn(*estimated.shape)
        
        # 归一化信号
        reference = reference / np.max(np.abs(reference))
        estimated = estimated / np.max(np.abs(estimated))
        
        # 重新排列维度以匹配museval的要求 [sources, samples, channels]
        reference = reference[np.newaxis, ...]  # [1, samples, channels]
        estimated = estimated[np.newaxis, ...]  # [1, samples, channels]
        
        try:
            # 使用museval库计算BSS指标
            sdr, isr, sir, sar = museval.evaluate(reference, estimated)
            
            # 处理无穷大和NaN值
            def handle_inf_nan(value):
                if np.isinf(value) or np.isnan(value):
                    return 0.0  # 如果是无穷大或NaN，返回0
                return float(value)
            
            # 取中值并处理异常值
            sdr_val = handle_inf_nan(np.nanmedian(sdr))
            sir_val = handle_inf_nan(np.nanmedian(sir))
            sar_val = handle_inf_nan(np.nanmedian(sar))
            isr_val = handle_inf_nan(np.nanmedian(isr))
            
            # 限制指标的合理范围
            def clip_metric(value, min_val=-100, max_val=100):
                return np.clip(value, min_val, max_val)
            
            return (
                clip_metric(sdr_val),
                clip_metric(sir_val),
                clip_metric(sar_val),
                clip_metric(isr_val)
            )
            
        except Exception as e:
            print(f"计算BSS指标时出错: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0
    
    def calculate_mse(self, reference: np.ndarray, estimated: np.ndarray) -> float:
        """
        计算均方误差(MSE)
        """
        return np.mean((reference - estimated) ** 2)
    
    def calculate_pesq(self, reference: np.ndarray, estimated: np.ndarray, sample_rate: int) -> float:
        """
        计算PESQ分数
        """
        try:
            # PESQ要求单声道音频
            if reference.ndim > 1:
                reference = np.mean(reference, axis=1)
            if estimated.ndim > 1:
                estimated = np.mean(estimated, axis=1)
                
            # 重采样到16kHz (PESQ要求)
            target_sr = 16000
            
            # 计算重采样比例
            ratio = target_sr / sample_rate
            
            # 计算新的采样点数
            new_length = int(len(reference) * ratio)
            
            # 重采样
            reference = signal.resample(reference, new_length)
            estimated = signal.resample(estimated, new_length)
            
            # 确保信号长度相同
            min_len = min(len(reference), len(estimated))
            reference = reference[:min_len]
            estimated = estimated[:min_len]
            
            # 计算PESQ分数
            score = pesq(target_sr, reference, estimated, 'wb')  # 'wb'表示宽带PESQ
            return score
        except Exception as e:
            print(f"PESQ计算错误: {e}")
            return 0.0
    
    def calculate_rtf(self, audio_length: float, processing_time: float) -> float:
        """
        计算实时因子(RTF)
        """
        return processing_time / audio_length
    
    def calculate_ldr(self, reference: np.ndarray, estimated: np.ndarray) -> float:
        """
        计算响度失真比(LDR)
        """
        try:
            # 计算响度
            def calculate_loudness(signal_data):
                if signal_data.ndim > 1:
                    # 如果是立体声，转换为单声道
                    signal_data = np.mean(signal_data, axis=1)
                
                # 使用RMS作为响度的简单度量
                rms = np.sqrt(np.mean(signal_data ** 2))
                return rms
                
            ref_loudness = calculate_loudness(reference)
            est_loudness = calculate_loudness(estimated)
            
            # 计算响度失真比
            return abs(20 * np.log10(est_loudness / (ref_loudness + 1e-8)))
        except Exception as e:
            print(f"LDR计算错误: {e}")
            return 0.0
    
    def evaluate_file(self, mixture_path: str, reference_path: str) -> Dict[str, float]:
        """
        评估单个音频文件
        """
        separated_files = []  # 用于记录生成的文件路径
        try:
            # 加载音频文件
            mixture, sr = sf.read(mixture_path)
            reference_solo, _ = sf.read(reference_path)
            
            # 统一截取前30秒进行评估
            max_length = 30 * sr  # 30秒
            if len(mixture) > max_length:
                mixture = mixture[:max_length]
                reference_solo = reference_solo[:max_length]
            
            # 使用模型分离音频
            try:
                start_time = time.time()
                separated_sources, _ = self.separator.separate_music_file(mixture, sr)
                processing_time = time.time() - start_time
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("GPU内存不足，尝试使用较短的音频段")
                    max_length = 15 * sr
                    mixture = mixture[:max_length]
                    reference_solo = reference_solo[:max_length]
                    torch.cuda.empty_cache()
                    start_time = time.time()
                    separated_sources, _ = self.separator.separate_music_file(mixture, sr)
                    processing_time = time.time() - start_time
                else:
                    raise e
            
            # 获取分离出的solo部分
            separated_solo = separated_sources['solo']
            
            # 保存分离结果
            try:
                output_dir = os.path.dirname(mixture_path)
                solo_path = os.path.join(output_dir, 'separated_solo.wav')
                other_path = os.path.join(output_dir, 'separated_other.wav')
                
                sf.write(solo_path, separated_solo, sr)
                sf.write(other_path, separated_sources['other'], sr)
                
                # 记录生成的文件路径
                separated_files.extend([solo_path, other_path])
                
            except Exception as e:
                print(f"保存分离结果时出错: {e}")
            
            # 确保信号长度相同
            min_len = min(len(reference_solo), len(separated_solo))
            ref = reference_solo[:min_len]
            est = separated_solo[:min_len]
            
            # 计算所有指标
            try:
                sdr, sir, sar, isr = self.calculate_sdr_sir_sar_isr(ref, est)
                mse = self.calculate_mse(ref, est)
                pesq_score = self.calculate_pesq(ref, est, sr)
                rtf = self.calculate_rtf(len(mixture)/sr, processing_time)
                ldr = self.calculate_ldr(ref, est)
                
                return {
                    'solo': {
                        'SDR': float(sdr),
                        'SIR': float(sir),
                        'SAR': float(sar),
                        'ISR': float(isr),
                        'MSE': float(mse),
                        'PESQ': float(pesq_score),
                        'RTF': float(rtf),
                        'LDR': float(ldr)
                    }
                }
            except Exception as e:
                print(f"计算评估指标时出错: {e}")
                return {
                    'solo': {
                        'SDR': 0.0,
                        'SIR': 0.0,
                        'SAR': 0.0,
                        'ISR': 0.0,
                        'MSE': 0.0,
                        'PESQ': 0.0,
                        'RTF': 0.0,
                        'LDR': 0.0
                    }
                }
            
        except Exception as e:
            print(f"评估文件时出错: {str(e)}")
            raise
        
        finally:
            # 删除生成的分离文件
            for file_path in separated_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"删除文件 {file_path} 时出错: {e}")
    
    def evaluate_dataset(self, dataset_path: str) -> Dict[str, Dict[str, float]]:
        """
        评估整个数据集
        """
        all_results = {}
        total_files = 0
        
        # 用于计算运行平均值的累加器
        metrics_sum = {
            'SDR': 0.0, 'SIR': 0.0, 'SAR': 0.0, 'ISR': 0.0,
            'MSE': 0.0, 'PESQ': 0.0, 'RTF': 0.0, 'LDR': 0.0
        }
        sample_count = 0
        
        # 首先计算总文件数
        for split in ['train', 'valid']:
            split_path = os.path.join(dataset_path, split)
            if os.path.isdir(split_path):
                for sample_dir in os.listdir(split_path):
                    if not sample_dir.startswith('.') and os.path.isdir(os.path.join(split_path, sample_dir)):
                        total_files += 1
        
        # 创建进度条
        pbar = tqdm(total=total_files, desc='评估进度', 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # 遍历train和valid目录
        for split in ['train', 'valid']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.isdir(split_path):
                continue
            
            # 遍历每个样本目录
            for sample_dir in os.listdir(split_path):
                if sample_dir.startswith('.'):  # 跳过.DS_Store等隐藏文件
                    continue
                
                sample_path = os.path.join(split_path, sample_dir)
                if os.path.isdir(sample_path):
                    mixture_path = os.path.join(sample_path, 'mixture.wav')
                    reference_path = os.path.join(sample_path, 'solo.wav')
                    
                    if os.path.exists(mixture_path) and os.path.exists(reference_path):
                        try:
                            results = self.evaluate_file(mixture_path, reference_path)
                            all_results[f"{split}/{sample_dir}"] = results
                            
                            # 更新累加器
                            sample_count += 1
                            for metric, value in results['solo'].items():
                                metrics_sum[metric] += value
                            
                            # 计算并显示当前平均值
                            print(f"\n当前评估指标均值 (已评估{sample_count}个样本):")
                            for metric in metrics_sum.keys():
                                current_avg = metrics_sum[metric] / sample_count
                                print(f"{metric}: {current_avg:.3f}")
                            
                        except Exception as e:
                            print(f"\n评估样本 {split}/{sample_dir} 时出错: {str(e)}")
                        finally:
                            pbar.update(1)  # 更新进度条
        
        pbar.close()  # 关闭进度条
        
        # 计算最终平均指标
        if all_results:
            avg_results = self.calculate_average_metrics(all_results)
            all_results['average'] = avg_results
        
        return all_results
    
    def calculate_average_metrics(self, results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        计算所有样本的平均指标
        """
        avg_results = {}
        
        # 初始化累加器
        metrics_sum = {}
        count = 0
        
        # 累加所有样本的指标
        for sample_results in results.values():
            for source_name, metrics in sample_results.items():
                if source_name not in metrics_sum:
                    metrics_sum[source_name] = {metric: 0.0 for metric in metrics}
                
                for metric, value in metrics.items():
                    metrics_sum[source_name][metric] += value
            count += 1
        
        # 计算平均值
        if count > 0:
            for source_name, metrics in metrics_sum.items():
                avg_results[source_name] = {
                    metric: value/count 
                    for metric, value in metrics.items()
                }
        
        return avg_results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估音源离模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config_path', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset_path', type=str, required=True, help='测试数据集路径')
    parser.add_argument('--output_path', type=str, default='evaluation_results.yaml', help='结果保存路径')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, args.config_path)
    
    # 评估数据集
    results = evaluator.evaluate_dataset(args.dataset_path)
    
    # 保存详细结果到yaml文件
    with open(args.output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # 只打印平均指标值
    avg_metrics = results['average']['solo']
    print("\n评估指标平均值:")
    print(f"SDR: {avg_metrics['SDR']:.3f}")
    print(f"SIR: {avg_metrics['SIR']:.3f}")
    print(f"SAR: {avg_metrics['SAR']:.3f}")
    print(f"ISR: {avg_metrics['ISR']:.3f}")
    print(f"MSE: {avg_metrics['MSE']:.3f}")
    print(f"PESQ: {avg_metrics['PESQ']:.3f}")
    print(f"RTF: {avg_metrics['RTF']:.3f}")
    print(f"LDR: {avg_metrics['LDR']:.3f}")

if __name__ == '__main__':
    main()
