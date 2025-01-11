from collections import defaultdict
from contextlib import contextmanager
import os
import tempfile
import typing as tp
import torch
import julius
from pathlib import Path
from contextlib import contextmanager

# Audio 音频处理相关函数
def convert_audio_channels(wav, channels=2):
    """
    将音频转换为指定的声道数。
    
    参数:
        wav: 输入的音频张量
        channels: 目标声道数，默认为2（立体声）
    
    返回:
        转换后的音频张量
    
    说明:
        - 如果输入是单声道，可以扩展到多声道
        - 如果输入是多声道，可以降为单声道（取平均）
        - 如果输入声道数大于目标声道数，截取前几个声道
        - 如果输入非单声道且声道数小于目标声道数，则报错
    """
    if wav.ndim == 1:
        src_channels = 1
    else:
        src_channels = wav.shape[-2]

    if src_channels == channels:
        pass
    elif channels == 1:
        if src_channels > 1:
            wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        wav = wav.expand(-1, channels, -1)
    elif src_channels >= channels:
        wav = wav[..., :channels, :]
    else:
        raise ValueError('输入音频的声道数小于目标声道数且不是单声道，无法进行转换。')
    return wav

def convert_audio(wav, from_samplerate, to_samplerate, channels):
    """
    转换音频的采样率和声道数。
    
    参数:
        wav: 输入的音频张量
        from_samplerate: 输入音频的采样率
        to_samplerate: 目标采样率
        channels: 目标声道数
    
    返回:
        转换后的音频张量
    
    说明:
        1. 首先转换声道数
        2. 然后进行采样率转换（如果需要）
    """
    wav = convert_audio_channels(wav, channels)
    
    if from_samplerate != to_samplerate:
        wav = julius.resample_frac(wav, from_samplerate, to_samplerate)
    return wav

def copy_state(state):
    """
    复制模型状态字典。
    
    参数:
        state: 模型状态字典
    
    返回:
        复制后的状态字典，所有张量都被移到CPU并创建副本
    """
    return {k: v.cpu().clone() for k, v in state.items()}

@contextmanager
def swap_state(model, state):
    """
    临时切换模型状态的上下文管理器。
    
    参数:
        model: 要切换状态的模型
        state: 要切换到的新状态
    
    使用示例:
        # 模型处于旧状态
        with swap_state(model, new_state):
            # 模型处于新状态
        # 模型恢复到旧状态
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)

@contextmanager
def temp_filenames(count: int, delete=True):
    """
    创建临时文件名的上下文管理器。
    
    参数:
        count: 需要创建的临时文件名数量
        delete: 退出上下文时是否删除临时文件，默认为True
    
    返回:
        临时文件名列表
    
    说明:
        在上下文结束时，如果delete=True，会自动删除创建的临时文件
    """
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)

def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    对张量进行中心裁剪，使其长度与参考张量相同。
    
    参数:
        tensor: 要裁剪的张量
        reference: 参考张量或目标长度
    
    返回:
        裁剪后的张量
    
    说明:
        - 如果长度差不是2的倍数，多出的样本会从右侧移除
        - 如果输入张量比参考短，会抛出ValueError
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("输入张量必须大于参考张量。差值为 {delta}。")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

def EMA(beta: float = 1):
    """
    指数移动平均（Exponential Moving Average）回调函数。
    
    参数:
        beta: 平滑系数，默认为1（此时相当于普通平均）
    
    返回:
        更新EMA的函数
    
    说明:
        - 用于跟踪训练过程中各项指标的移动平均值
        - beta=1时退化为普通的算术平均
        - 可以通过返回的函数持续更新多个指标的移动平均
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update

class DummyPoolExecutor:
    """
    线程池执行器的简单实现。
    
    说明:
        - 用于在不需要真正并行处理时模拟线程池的行为
        - 主要用于测试或单线程环境
    """
    class DummyResult:
        """
        模拟Future对象的结果类。
        """
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            """
            执行函数并返回结果。
            """
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        """
        初始化虚拟线程池。
        
        参数:
            workers: 工作线程数（在此实现中被忽略）
        """
        pass

    def submit(self, func, *args, **kwargs):
        """
        提交任务到虚拟线程池。
        """
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return

def new_sdr(references, estimates):
    """
    计算信号失真比（Signal to Distortion Ratio, SDR）。
    
    参数:
        references: 参考信号，形状为 [批次, 源数量, 声道数, 采样点数]
        estimates: 估计信号，形状同上
    
    返回:
        SDR分数，单位为分贝(dB)
    
    说明:
        - SDR越高表示重建质量越好
        - 使用公式：SDR = 10 * log10(参考信号能量 / 误差信号能量)
        - 添加小的delta值以避免数值问题
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # 避免数值误差
    num = torch.sum(torch.square(references), dim=(2, 3))
    den = torch.sum(torch.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

def load_model(model, checkpoint_path):
    """
    加载模型检查点。
    
    参数:
        model: 要加载权重的模型
        checkpoint_path: 检查点文件路径
        
    返回:
        加载了权重的模型
        
    说明:
        支持多种检查点格式：
        1. 包含 'state' 的字典（最新状态）
        2. 包含 'best_state' 的字典（最佳状态）
        3. 包含 'state_dict' 的字典
        4. 直接的状态字典
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型检查点文件：{checkpoint_path}")
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        raise RuntimeError(f"加载检查点文件时出错：{e}")
    
    if checkpoint is None:
        raise ValueError(f"检查点文件 {checkpoint_path} 加载后为空")
    
    # 尝试不同的键名获取状态字典
    state_dict = None
    if isinstance(checkpoint, dict):
        # 按优先级尝试不同的键，现在优先使用 state
        for key in ['state', 'best_state', 'state_dict']:
            if key in checkpoint and checkpoint[key] is not None:
                state_dict = checkpoint[key]
                print(f"使用检查点中的 '{key}' 键")
                break
        # 如果没有找到有效的键，使用整个检查点
        if state_dict is None:
            state_dict = checkpoint
            print("使用整个检查点作为状态字典")
    else:
        state_dict = checkpoint
        print("检查点不是字典类型，直接作为状态字典使用")
    
    if state_dict is None:
        raise ValueError("无法从检查点中提取有效的状态字典")
        
    # 处理多 GPU 训练产生的 'module.' 前缀
    new_state_dict = {}
    try:
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
    except Exception as e:
        raise RuntimeError(f"处理状态字典时出错：{e}")
            
    try:
        model.load_state_dict(new_state_dict)
        print("模型权重加载成功")
    except Exception as e:
        print(f"加载状态字典时出错：{e}")
        print("检查点中可用的键：", list(state_dict.keys()))
        print("模型需要的键：", list(model.state_dict().keys()))
        raise
        
    return model


