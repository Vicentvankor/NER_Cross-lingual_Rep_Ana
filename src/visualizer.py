"""
可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """可视化器类"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        plt.style.use('default')
        
    def plot_tsne(
        self, 
        features: np.ndarray, 
        languages: List[str],
        title: str = "NER Input Representation, different languages",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        使用t-SNE进行降维可视化
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            matplotlib图表对象
        """
        logger.info("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=self.random_seed, perplexity=min(30, len(features)-1))
        embed2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取唯一语言并分配颜色
        unique_langs = sorted(list(set(languages)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_langs)))
        
        # 绘制散点图
        for color, lang in zip(colors, unique_langs):
            indices = [i for i, l in enumerate(languages) if l == lang]
            ax.scatter(
                embed2d[indices, 0], 
                embed2d[indices, 1], 
                label=lang, 
                color=color,
                alpha=0.7,
                s=50
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"t-SNE plot saved to {save_path}")
            
        return fig
    
    def plot_pca(
        self, 
        features: np.ndarray, 
        languages: List[str],
        title: str = "PCA of NER Input Representations",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        使用PCA进行降维可视化
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            matplotlib图表对象
        """
        logger.info("Performing PCA dimensionality reduction...")
        pca = PCA(n_components=2, random_state=self.random_seed)
        embed2d = pca.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取唯一语言并分配颜色
        unique_langs = sorted(list(set(languages)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_langs)))
        
        # 绘制散点图
        for color, lang in zip(colors, unique_langs):
            indices = [i for i, l in enumerate(languages) if l == lang]
            ax.scatter(
                embed2d[indices, 0], 
                embed2d[indices, 1], 
                label=lang, 
                color=color,
                alpha=0.7,
                s=50
            )
        
        # 添加方差贡献率信息
        variance_ratio = pca.explained_variance_ratio_
        ax.set_title(f"{title}\n(PC1: {variance_ratio[0]:.1%}, PC2: {variance_ratio[1]:.1%})", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} variance)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PCA plot saved to {save_path}")
            
        return fig
    
    def plot_language_similarity_matrix(
        self, 
        features: np.ndarray, 
        languages: List[str],
        title: str = "Cross-lingual Similarity Matrix",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        绘制语言间相似性矩阵热图
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            matplotlib图表对象
        """
        logger.info("Computing cross-lingual similarity matrix...")
        
        # 计算每种语言的平均特征向量
        unique_langs = sorted(list(set(languages)))
        lang_centroids = {}
        
        for lang in unique_langs:
            indices = [i for i, l in enumerate(languages) if l == lang]
            lang_centroids[lang] = features[indices].mean(axis=0)
        
        # 构建相似性矩阵
        centroid_matrix = np.array([lang_centroids[lang] for lang in unique_langs])
        similarity_matrix = cosine_similarity(centroid_matrix)
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(unique_langs)))
        ax.set_yticks(range(len(unique_langs)))
        ax.set_xticklabels(unique_langs)
        ax.set_yticklabels(unique_langs)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(unique_langs)):
            for j in range(len(unique_langs)):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white" if similarity_matrix[i, j] < 0.5 else "black")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cosine Similarity', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Similarity matrix saved to {save_path}")
            
        return fig
    
    def plot_language_distribution(
        self, 
        languages: List[str],
        title: str = "Language Distribution in Dataset",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        绘制语言分布柱状图
        
        Args:
            languages: 语言标签列表
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
            
        Returns:
            matplotlib图表对象
        """
        # 统计语言分布
        lang_counts = {}
        for lang in languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # 按语言名排序
        sorted_langs = sorted(lang_counts.keys())
        counts = [lang_counts[lang] for lang in sorted_langs]
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(sorted_langs, counts, color=plt.cm.tab10(np.linspace(0, 1, len(sorted_langs))))
        
        # 添加数值标注
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Language distribution plot saved to {save_path}")
            
        return fig
    
    def create_analysis_report(
        self, 
        features: np.ndarray, 
        languages: List[str],
        output_dir: Path,
        experiment_name: str = "base_model"
    ) -> Dict[str, Path]:
        """
        创建完整的可视化分析报告
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            output_dir: 输出目录
            experiment_name: 实验名称
            
        Returns:
            保存的文件路径字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        # 1. t-SNE可视化
        tsne_path = output_dir / f"{experiment_name}_tsne.png"
        self.plot_tsne(features, languages, save_path=tsne_path)
        saved_plots['tsne'] = tsne_path
        plt.close()
        
        # 2. PCA可视化
        pca_path = output_dir / f"{experiment_name}_pca.png"
        self.plot_pca(features, languages, save_path=pca_path)
        saved_plots['pca'] = pca_path
        plt.close()
        
        # 3. 语言相似性矩阵
        similarity_path = output_dir / f"{experiment_name}_similarity_matrix.png"
        self.plot_language_similarity_matrix(features, languages, save_path=similarity_path)
        saved_plots['similarity_matrix'] = similarity_path
        plt.close()
        
        # 4. 语言分布
        distribution_path = output_dir / f"{experiment_name}_language_distribution.png"
        self.plot_language_distribution(languages, save_path=distribution_path)
        saved_plots['language_distribution'] = distribution_path
        plt.close()
        
        logger.info(f"Analysis report created in {output_dir}")
        for plot_type, path in saved_plots.items():
            logger.info(f"  {plot_type}: {path}")
            
        return saved_plots