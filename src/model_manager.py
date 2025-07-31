"""
模型管理模块
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging

from config.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._is_loaded = False
        
    def load_model(self, lora_path: Optional[str] = None) -> None:
        """
        加载模型
        
        Args:
            lora_path: LoRA模型路径，如果为None则只加载基础模型
        """
        try:
            logger.info(f"Loading tokenizer from {self.config.base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model, 
                trust_remote_code=self.config.trust_remote_code
            )
            
            logger.info(f"Loading base model from {self.config.base_model}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                trust_remote_code=self.config.trust_remote_code,
                device_map=self.config.device_map,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            )
            
            if lora_path and Path(lora_path).exists():
                logger.info(f"Loading LoRA model from {lora_path}")
                self.model = PeftModel.from_pretrained(base_model, lora_path)
                logger.info("LoRA model loaded successfully")
            else:
                self.model = base_model
                if lora_path:
                    logger.warning(f"LoRA path {lora_path} not found, using base model only")
                else:
                    logger.info("Using base model without LoRA")
                    
            self.model.eval()
            self._is_loaded = True
            logger.info("Model loaded and set to evaluation mode")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded and self.model is not None and self.tokenizer is not None
    
    def get_input_hidden_vector(
        self, 
        prompt: str, 
        input_text: str, 
        layer_idx: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        提取输入文本的隐藏层表示向量
        
        Args:
            prompt: 提示词
            input_text: 输入文本
            layer_idx: 层索引，默认使用配置中的值
            
        Returns:
            隐藏层向量或None（如果出错）
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if layer_idx is None:
            layer_idx = self.config.layer_idx
            
        try:
            full_input = prompt + input_text
            
            # 获取提示词和完整输入的token IDs
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            full_ids = self.tokenizer(full_input, add_special_tokens=False)['input_ids']
            
            # 计算输入文本对应的token索引
            input_token_indices = list(range(len(prompt_ids), len(full_ids)))
            
            if len(input_token_indices) == 0:
                logger.warning(f"No input tokens detected for text: {input_text}")
                return None
            
            # 编码并移到设备
            encoded = self.tokenizer(full_input, return_tensors='pt')
            encoded = {k: v.to(self.config.device) for k, v in encoded.items()}
            
            # 前向传播获取隐藏状态
            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
            
            # 提取输入文本对应的隐藏状态
            input_token_hiddens = hidden[input_token_indices, :]  # [input_len, hidden_dim]
            
            # 池化操作（平均）
            pooled = input_token_hiddens.mean(dim=0)
            
            return pooled.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting hidden vector for text '{input_text}': {e}")
            return None
    
    def batch_extract_features(
        self, 
        samples: List[Dict[str, Any]], 
        prompt: str,
        layer_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        批量提取特征
        
        Args:
            samples: 样本列表，每个样本包含'text'和'language'字段
            prompt: 提示词
            layer_idx: 层索引
            
        Returns:
            特征矩阵和语言列表的元组
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        features = []
        languages = []
        
        logger.info(f"Extracting features for {len(samples)} samples")
        
        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing sample {i + 1}/{len(samples)}")
                
            text = sample.get('text', '')
            language = sample.get('language', 'unknown')
            
            if not text.strip():
                logger.warning(f"Empty text in sample {i}")
                continue
                
            vector = self.get_input_hidden_vector(prompt, text, layer_idx)
            
            if vector is not None:
                features.append(vector)
                languages.append(language)
            else:
                logger.warning(f"Failed to extract features for sample {i}: {text[:50]}...")
        
        if not features:
            raise ValueError("No valid features extracted from samples")
            
        features_array = np.vstack(features)
        logger.info(f"Extracted features shape: {features_array.shape}")
        
        return features_array, languages
    
    def unload_model(self) -> None:
        """卸载模型以释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer  
            self.tokenizer = None
        self._is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Model unloaded and GPU memory cleared")