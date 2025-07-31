"""
主程序入口
"""
import argparse
import logging
from pathlib import Path
import random
import numpy as np
import torch

from config.model_config import model_config, experiment_config, prompt_config
from src.data_loader import DataLoader
from src.model_manager import ModelManager
from src.feature_extractor import FeatureExtractor
from src.visualizer import Visualizer
from src.analyzer import CrossLingualAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_random_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

class NERCrossLingualAnalysis:
    """NER跨语言分析主类"""
    
    def __init__(self, args):
        self.args = args
        self.setup_directories()
        
        # 初始化组件，使用配置中的文件路径模板
        self.data_loader = DataLoader(
            data_dir=experiment_config.data_dir,
            supported_languages=experiment_config.supported_languages,
            file_path_template=experiment_config.file_path_template
        )
        self.model_manager = ModelManager(model_config)
        self.feature_extractor = None  # 将在模型加载后初始化
        self.visualizer = Visualizer(random_seed=experiment_config.random_seed)
        self.analyzer = CrossLingualAnalyzer(random_seed=experiment_config.random_seed)
        
    def setup_directories(self):
        """创建必要的目录"""
        for dir_name in [experiment_config.output_dir, experiment_config.results_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
    def load_data(self):
        """加载数据"""
        logger.info("Loading data...")
        
        if self.args.use_real_data:
            # 加载真实数据
            samples = self.data_loader.load_retri_data(
                split="train", 
                max_samples_per_lang=experiment_config.max_samples_per_lang
            )
            if not samples:
                logger.warning("No real data found, falling back to test samples")
                samples = self.data_loader.create_test_samples()
        else:
            # 使用测试数据
            samples = self.data_loader.create_test_samples()
            
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def run_base_model_analysis(self):
        """运行基础模型分析"""
        logger.info("Starting base model analysis...")
        
        # 设置随机种子
        set_random_seed(experiment_config.random_seed)
        
        # 加载数据
        samples = self.load_data()
        
        # 加载模型（不使用LoRA）
        logger.info("Loading base model...")
        self.model_manager.load_model(lora_path=None)
        self.feature_extractor = FeatureExtractor(self.model_manager, prompt_config)
        
        # 提取特征
        experiment_name = "base_model"
        output_dir = Path(experiment_config.output_dir)
        
        logger.info("Extracting features...")
        saved_files = self.feature_extractor.extract_and_save(
            samples=samples,
            output_dir=output_dir,
            experiment_name=experiment_name
        )
        
        # 加载提取的特征进行分析
        features, languages = self.feature_extractor.load_features(
            saved_files['features'], 
            saved_files['languages']
        )
        
        # 生成可视化报告
        logger.info("Generating visualizations...")
        viz_output_dir = Path(experiment_config.results_dir) / "visualizations"
        self.visualizer.create_analysis_report(
            features=features,
            languages=languages,
            output_dir=viz_output_dir,
            experiment_name=experiment_name
        )
        
        # 生成分析报告
        logger.info("Generating analysis report...")
        analysis_report = self.analyzer.generate_analysis_report(features, languages)
        
        # 保存分析报告
        report_path = Path(experiment_config.results_dir) / f"{experiment_name}_analysis_report.json"
        self.analyzer.save_analysis_report(analysis_report, str(report_path))
        
        # 卸载模型释放内存
        self.model_manager.unload_model()
        
        logger.info("Base model analysis completed!")
        return analysis_report
    
    def run_lora_comparison_analysis(self, lora_paths: dict):
        """运行LoRA对比分析（未来扩展用）"""
        logger.info("LoRA comparison analysis is not implemented yet.")
        logger.info("This feature will be available when you have trained LoRA models.")
        
        # 为未来扩展预留的接口
        for model_name, lora_path in lora_paths.items():
            logger.info(f"Would analyze model: {model_name} with LoRA path: {lora_path}")
            
    def print_summary(self, analysis_report):
        """打印分析总结"""
        print("\n" + "="*60)
        print("NER Cross-lingual Representation Analysis Summary")
        print("="*60)
        
        dataset_info = analysis_report['dataset_info']
        print(f"Dataset Information:")
        print(f"  Total samples: {dataset_info['total_samples']}")
        print(f"  Feature dimension: {dataset_info['feature_dimension']}")
        print(f"  Languages: {', '.join(dataset_info['languages'])}")
        print(f"  Samples per language: {dataset_info['samples_per_language']}")
        
        similarity_info = analysis_report['cross_lingual_similarity']
        print(f"\nCross-lingual Similarity:")
        print(f"  Average similarity: {similarity_info['avg_similarity']:.4f}")
        print(f"  Most similar pair: {similarity_info['most_similar_pair']} ({similarity_info['most_similar_score']:.4f})")
        print(f"  Least similar pair: {similarity_info['least_similar_pair']} ({similarity_info['least_similar_score']:.4f})")
        
        distance_info = analysis_report['distance_comparison']
        print(f"\nDistance Analysis:")
        print(f"  Intra-language distance: {distance_info['intra_language_distances']['mean']:.4f} ± {distance_info['intra_language_distances']['std']:.4f}")
        print(f"  Inter-language distance: {distance_info['inter_language_distances']['mean']:.4f} ± {distance_info['inter_language_distances']['std']:.4f}")
        print(f"  Statistical significance: {'Yes' if distance_info['statistical_test']['significant'] else 'No'} (p={distance_info['statistical_test']['p_value']:.4f})")
        
        print(f"\nResult files saved in: {experiment_config.results_dir}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="NER Cross-lingual Representation Analysis")
    parser.add_argument("--use-real-data", action="store_true", 
                       help="Use real data from Retri_data instead of test samples")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Override base model path")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum samples per language")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for features")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory for analysis outputs")
    
    args = parser.parse_args()
    
    # 更新配置
    if args.base_model:
        model_config.base_model = args.base_model
    if args.max_samples:
        experiment_config.max_samples_per_lang = args.max_samples
    if args.output_dir:
        experiment_config.output_dir = args.output_dir
    if args.results_dir:
        experiment_config.results_dir = args.results_dir
    
    # 创建分析器实例
    analyzer = NERCrossLingualAnalysis(args)
    
    try:
        # 运行基础模型分析
        analysis_report = analyzer.run_base_model_analysis()
        
        # 打印总结
        analyzer.print_summary(analysis_report)
        
        # 如果有LoRA模型路径，运行对比分析
        # 这里可以在未来添加LoRA路径的处理逻辑
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()