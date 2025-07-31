"""
数据加载模块

该模块负责从各种数据源加载多语言NER样本，包括：
1. 从JSONL格式的Retri数据集加载真实样本
2. 创建测试样本用于快速验证
3. 统一的样本数据结构管理

支持的数据格式：
- JSONL文件：每行一个JSON对象，包含instruction、input、output字段
- 自动样本数量控制和随机采样
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Sample:
    """
    单个样本的数据结构
    
    统一管理从不同数据源加载的样本信息
    """
    # 原始文本内容，用于特征提取
    text: str
    # 语言代码（如：'en', 'zh', 'de'等）
    language: str
    # 可选的实体列表，用于后续分析
    entities: Optional[List[str]] = None
    # 可选的实体类别列表
    categories: Optional[List[str]] = None
    # 可选的原始指令
    instruction: Optional[str] = None
    # 可选的期望输出
    expected_output: Optional[str] = None

class DataLoader:
    """
    数据加载器类
    
    负责从不同数据源加载和预处理多语言NER样本
    """
    
    def __init__(self, data_dir: str, supported_languages: List[str], file_path_template: str = "Retri_{split}_{lang}.jsonl"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据根目录路径
            supported_languages: 支持的语言列表
            file_path_template: 文件路径模板，支持{split}和{lang}占位符
        """
        self.data_dir = Path(data_dir)
        self.supported_languages = supported_languages
        self.file_path_template = file_path_template
        
    def load_retri_data(self, split: str = "train", max_samples_per_lang: int = 100) -> List[Sample]:
        """
        加载Retri数据集
        
        从指定目录加载各语言的JSONL格式数据文件
        
        Args:
            split: 数据集分割（train/test）
            max_samples_per_lang: 每种语言最大样本数，用于控制内存使用
            
        Returns:
            样本列表，包含所有语言的混合数据
        """
        samples = []
        
        for lang in self.supported_languages:
            # 使用配置的文件路径模板构建文件路径
            filename = self.file_path_template.format(split=split, lang=lang)
            file_path = self.data_dir / split / filename
            
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping {lang}")
                continue
                
            lang_samples = self._load_jsonl_file(file_path, lang, max_samples_per_lang)
            samples.extend(lang_samples)
            print(f"Loaded {len(lang_samples)} samples for {lang}")
            
        return samples
    
    def _load_jsonl_file(self, file_path: Path, language: str, max_samples: int) -> List[Sample]:
        """
        加载单个JSONL文件
        
        解析每行的JSON数据，提取文本内容和相关信息
        
        Args:
            file_path: JSONL文件路径
            language: 语言代码
            max_samples: 最大样本数
            
        Returns:
            该语言的样本列表
        """
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 随机采样以确保数据多样性，避免只取前N个样本
            if len(lines) > max_samples:
                random.seed(42)  # 固定随机种子确保可重现
                lines = random.sample(lines, max_samples)
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # 解析JSON数据
                    data = json.loads(line)
                    
                    # 提取文本内容和相关信息
                    text = self._extract_text_from_data(data)
                    if text:
                        # 解析实体和类别信息
                        entities, categories = self._extract_entities_from_output(data.get('output', ''))
                        
                        sample = Sample(
                            text=text,
                            language=language,
                            entities=entities,
                            categories=categories,
                            instruction=data.get('instruction'),
                            expected_output=data.get('output')
                        )
                        samples.append(sample)
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in {file_path} line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num} in {file_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return samples
    
    def _extract_text_from_data(self, data: Dict[str, Any]) -> Optional[str]:
        """
        从JSON数据中提取文本内容
        
        根据Retri数据集的格式，优先从'input'字段提取文本
        
        Args:
            data: 原始JSON数据字典
            
        Returns:
            提取的文本内容或None
        """
        # Retri数据集的标准格式：input字段包含待分析的文本
        if 'input' in data and data['input']:
            text = data['input'].strip()
            if text:
                return text
        
        # 备用字段，兼容其他可能的格式
        backup_fields = ['text', 'sentence', 'content', 'query']
        for field in backup_fields:
            if field in data and data[field]:
                text = data[field].strip()
                if text:
                    return text
                    
        # 处理对话格式数据（如果存在）
        if 'conversations' in data:
            conversations = data['conversations']
            if isinstance(conversations, list) and len(conversations) > 0:
                for conv in conversations:
                    if conv.get('from') == 'human' and conv.get('value'):
                        return conv['value'].strip()
                        
        return None
    
    def _extract_entities_from_output(self, output_str: str) -> tuple[Optional[List[str]], Optional[List[str]]]:
        """
        从输出字符串中解析实体和类别信息
        
        解析JSON格式的输出，提取实体列表和对应的类别
        
        Args:
            output_str: 输出字符串，通常是JSON格式
            
        Returns:
            (实体列表, 类别列表) 的元组
        """
        if not output_str:
            return None, None
            
        try:
            # 解析JSON格式的输出
            output_data = json.loads(output_str)
            if not isinstance(output_data, dict):
                return None, None
                
            entities = []
            categories = []
            
            # 遍历每个实体类型
            for entity_type, entity_list in output_data.items():
                if isinstance(entity_list, list):
                    entities.extend(entity_list)
                    # 为每个实体添加对应的类别
                    categories.extend([entity_type] * len(entity_list))
            
            return entities if entities else None, categories if categories else None
            
        except json.JSONDecodeError:
            # 如果不是有效的JSON，返回None
            return None, None
        except Exception:
            return None, None
    
    def create_test_samples(self) -> List[Sample]:
        """
        创建测试样本（用于演示和快速验证）
        
        生成包含多种语言的示例数据，用于测试模型和可视化功能
        
        Returns:
            测试样本列表，包含8种语言的典型NER文本
        """
        test_data = [
            {"lang": "zh", "text": "北見工業大學位于日本北海道，是一所著名的理工科大学"},
            {"lang": "en", "text": "The University of Tokyo is located in Japan and is one of the most prestigious universities in Asia."},
            {"lang": "ja", "text": "北海道大学は札幌にあり、日本の国立大学の一つです。"},
            {"lang": "de", "text": "Die Universität München ist eine der führenden Hochschulen in Deutschland."},
            {"lang": "fr", "text": "L'Université de Paris est située en France et offre de nombreux programmes d'études."},
            {"lang": "es", "text": "La Universidad de Barcelona está en España y es conocida por su excelencia académica."},
            {"lang": "ko", "text": "서울대학교는 한국의 명문대학으로 많은 유명한 졸업생을 배출했습니다."},
            {"lang": "ru", "text": "Московский государственный университет находится в России и является ведущим учебным заведением."}
        ]
        
        return [Sample(text=item["text"], language=item["lang"]) for item in test_data]