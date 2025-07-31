"""
模型配置文件

该模块包含了NER跨语言表示分析项目的所有配置类，用于统一管理：
1. 模型相关配置（ModelConfig）
2. 实验流程配置（ExperimentConfig）  
3. 提示词模板配置（PromptConfig）

配置采用dataclass设计，便于类型检查和默认值管理。
"""
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """
    模型配置类
    
    管理与模型加载、推理相关的所有参数配置
    """
    # 基础模型路径，支持HuggingFace模型名或本地路径
    base_model: str = "Qwen/Qwen2.5-7B"
    
    # LoRA模型路径，可选参数，用于加载微调后的适配器权重
    # 如果为None，则只使用基础模型进行分析
    lora_model: Optional[str] = None
    
    # 推理设备，自动检测GPU可用性，优先使用CUDA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 提取隐藏层的索引，-2表示倒数第二层
    # 经验上倒数第二层包含更丰富的语义信息，适合表示学习分析
    layer_idx: int = -2
    
    # 是否信任远程代码，某些模型（如Qwen）需要此参数
    trust_remote_code: bool = True
    
    # 设备映射策略，'auto'让transformers自动分配GPU内存
    device_map: str = 'auto'

@dataclass
class ExperimentConfig:
    """
    实验配置类
    
    管理数据加载、输出路径、实验参数等配置
    """
    # 数据根目录，包含各语言的训练/测试数据
    # 更新为正确的路径：data/Retri_data
    data_dir: str = "data/Retri_data"
    
    # 特征输出目录，用于保存提取的隐藏层表示
    output_dir: str = "outputs"
    
    # 分析结果目录，用于保存可视化图表和分析报告
    results_dir: str = "results"
    
    # 支持的语言列表，None表示使用默认的8种语言
    supported_languages: List[str] = None
    
    # 每种语言的最大样本数，用于控制实验规模和内存使用
    max_samples_per_lang: int = 100
    
    # 随机种子，确保实验结果可复现
    random_seed: int = 42
    
    # 数据文件路径模板配置
    # 用于构建各语言数据文件的路径，{split}为train/test，{lang}为语言代码
    file_path_template: str = "Retri_{split}_{lang}.jsonl"
    
    # 支持的数据集分割类型
    supported_splits: List[str] = None
    
    def __post_init__(self):
        """
        初始化后处理，设置默认支持的语言列表和分割类型
        包含德语、英语、西班牙语、法语、日语、韩语、俄语、中文
        """
        if self.supported_languages is None:
            self.supported_languages = ['de', 'en', 'es', 'fr', 'ja', 'ko', 'ru', 'zh']
        
        if self.supported_splits is None:
            self.supported_splits = ['train', 'test']

@dataclass
class PromptConfig:
    """
    提示词配置类（可选覆盖配置）
    
    注意：当前JSONL数据文件中每个样本都已包含完整的instruction字段，
    因此此配置类主要用于以下场景：
    1. 需要统一覆盖所有样本的指令时
    2. 处理没有instruction字段的数据源时
    3. 进行实验对比时使用不同的提示词模板
    """
    
    # 是否使用自定义提示词覆盖数据中的instruction字段
    # False: 使用数据中的instruction字段（推荐）
    # True: 使用下面的custom_prompt覆盖
    use_custom_prompt: bool = False
    
    # 自定义NER任务提示词模板（仅在use_custom_prompt=True时使用）
    # 当前数据中的instruction已经包含了完整的NER指令，通常不需要覆盖
    custom_prompt: str = (
        "Please list all named entities of the following entity types in the input sentence:"
        "\n- PERSON - LOCATION - PRODUCT - FACILITY - ART - GROUP - MISCELLANEOUS - SCIENCE ENTITY:"
        "You should output your results in the format {\"type\": [\"entity\"]} as a JSON.\nInput: "
    )
    
    # 备用提示词，用于测试样本（没有instruction字段的情况）
    fallback_prompt: str = (
        "Please list all named entities of the following entity types in the input sentence:"
        "\n- PERSON - LOCATION - PRODUCT - FACILITY - ART - GROUP - MISCELLANEOUS - SCIENCE ENTITY:"
        "You should output your results in the format {\"type\": [\"entity\"]} as a JSON.\nInput: "
    )

# 全局配置实例
# 这些实例将在整个项目中被导入和使用，提供统一的配置访问点
model_config = ModelConfig()        # 模型配置实例
experiment_config = ExperimentConfig()  # 实验配置实例  
prompt_config = PromptConfig()      # 提示词配置实例