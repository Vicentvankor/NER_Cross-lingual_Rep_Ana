"""
特征提取模块
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

from src.model_manager import ModelManager
from src.data_loader import Sample
from config.model_config import PromptConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """特征提取器类"""
    
    def __init__(self, model_manager: ModelManager, prompt_config: PromptConfig):
        self.model_manager = model_manager
        self.prompt_config = prompt_config
        
    def _get_prompt_for_sample(self, sample: Sample) -> str:
        """
        为样本获取合适的提示词
        
        Args:
            sample: 样本对象
            
        Returns:
            提示词字符串
        """
        # 如果配置要求使用自定义提示词，则覆盖数据中的instruction
        if self.prompt_config.use_custom_prompt:
            return self.prompt_config.custom_prompt
        
        # 优先使用样本自带的instruction（推荐方式）
        if sample.instruction:
            return sample.instruction
        
        # 如果样本没有instruction，使用备用提示词
        logger.warning(f"Sample has no instruction, using fallback prompt")
        return self.prompt_config.fallback_prompt
        
    def extract_features_from_samples(
        self, 
        samples: List[Sample], 
        layer_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        从样本中提取特征
        
        Args:
            samples: 样本列表
            layer_idx: 层索引
            
        Returns:
            (特征矩阵, 语言列表, 文本列表)
        """
        if not self.model_manager.is_loaded():
            raise RuntimeError("Model not loaded")
            
        features = []
        languages = []
        texts = []
        
        logger.info(f"Extracting features from {len(samples)} samples")
        
        # 统计使用的提示词来源
        instruction_sources = {"data_instruction": 0, "custom_prompt": 0, "fallback_prompt": 0}
        
        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing sample {i + 1}/{len(samples)}")
                
            if not sample.text.strip():
                logger.warning(f"Empty text in sample {i}")
                continue
            
            # 智能选择提示词
            prompt = self._get_prompt_for_sample(sample)
            
            # 统计提示词来源
            if self.prompt_config.use_custom_prompt:
                instruction_sources["custom_prompt"] += 1
            elif sample.instruction:
                instruction_sources["data_instruction"] += 1
            else:
                instruction_sources["fallback_prompt"] += 1
                
            vector = self.model_manager.get_input_hidden_vector(
                prompt, 
                sample.text, 
                layer_idx
            )
            
            if vector is not None:
                features.append(vector)
                languages.append(sample.language)
                texts.append(sample.text)
            else:
                logger.warning(f"Failed to extract features for sample {i}: {sample.text[:50]}...")
        
        # 输出提示词使用统计
        logger.info("Prompt source statistics:")
        for source, count in instruction_sources.items():
            if count > 0:
                logger.info(f"  {source}: {count}")
        
        if not features:
            raise ValueError("No valid features extracted from samples")
            
        features_array = np.vstack(features)
        logger.info(f"Extracted features shape: {features_array.shape}")
        
        return features_array, languages, texts
    
    def save_features(
        self, 
        features: np.ndarray, 
        languages: List[str], 
        texts: List[str], 
        output_dir: Path,
        experiment_name: str = "base_model"
    ) -> Dict[str, Path]:
        """
        保存提取的特征
        
        Args:
            features: 特征矩阵
            languages: 语言列表
            texts: 文本列表
            output_dir: 输出目录
            experiment_name: 实验名称
            
        Returns:
            保存的文件路径字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存特征矩阵
        features_path = output_dir / f"{experiment_name}_features.npy"
        np.save(features_path, features)
        
        # 保存语言标签
        languages_path = output_dir / f"{experiment_name}_languages.txt"
        with open(languages_path, 'w', encoding='utf-8') as f:
            for lang in languages:
                f.write(f"{lang}\n")
        
        # 保存文本内容
        texts_path = output_dir / f"{experiment_name}_texts.txt"
        with open(texts_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(f"{text}\n")
        
        # 保存元数据
        metadata_path = output_dir / f"{experiment_name}_metadata.txt"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Features shape: {features.shape}\n")
            f.write(f"Number of samples: {len(languages)}\n")
            f.write(f"Languages: {sorted(set(languages))}\n")
            f.write(f"Samples per language:\n")
            for lang in sorted(set(languages)):
                count = languages.count(lang)
                f.write(f"  {lang}: {count}\n")
        
        saved_files = {
            'features': features_path,
            'languages': languages_path,
            'texts': texts_path,
            'metadata': metadata_path
        }
        
        logger.info(f"Features saved to {output_dir}")
        for key, path in saved_files.items():
            logger.info(f"  {key}: {path}")
            
        return saved_files
    
    def load_features(self, features_path: Path, languages_path: Path) -> Tuple[np.ndarray, List[str]]:
        """
        加载保存的特征
        
        Args:
            features_path: 特征文件路径
            languages_path: 语言标签文件路径
            
        Returns:
            (特征矩阵, 语言列表)
        """
        features = np.load(features_path)
        
        with open(languages_path, 'r', encoding='utf-8') as f:
            languages = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Loaded features shape: {features.shape}")
        logger.info(f"Loaded {len(languages)} language labels")
        
        return features, languages
    
    def extract_and_save(
        self, 
        samples: List[Sample], 
        output_dir: Path,
        experiment_name: str = "base_model",
        layer_idx: Optional[int] = None
    ) -> Dict[str, Path]:
        """
        提取特征并保存
        
        Args:
            samples: 样本列表
            output_dir: 输出目录
            experiment_name: 实验名称
            layer_idx: 层索引
            
        Returns:
            保存的文件路径字典
        """
        features, languages, texts = self.extract_features_from_samples(samples, layer_idx)
        return self.save_features(features, languages, texts, output_dir, experiment_name)