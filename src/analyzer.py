"""
分析模块
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu, ttest_ind
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossLingualAnalyzer:
    """跨语言分析器类"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        
    def compute_language_centroids(self, features: np.ndarray, languages: List[str]) -> Dict[str, np.ndarray]:
        """
        计算每种语言的中心向量
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            
        Returns:
            语言中心向量字典
        """
        unique_langs = sorted(set(languages))
        centroids = {}
        
        for lang in unique_langs:
            indices = [i for i, l in enumerate(languages) if l == lang]
            centroids[lang] = features[indices].mean(axis=0)
            
        return centroids
    
    def compute_cross_lingual_similarity(self, features: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """
        计算跨语言相似性指标
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            
        Returns:
            相似性分析结果
        """
        # 计算语言中心向量
        centroids = self.compute_language_centroids(features, languages)
        unique_langs = sorted(centroids.keys())
        
        # 计算语言间相似性矩阵
        similarity_matrix = np.zeros((len(unique_langs), len(unique_langs)))
        for i, lang1 in enumerate(unique_langs):
            for j, lang2 in enumerate(unique_langs):
                sim = cosine_similarity([centroids[lang1]], [centroids[lang2]])[0, 0]
                similarity_matrix[i, j] = sim
        
        # 计算平均相似性指标
        upper_triangle = np.triu(similarity_matrix, k=1)
        avg_similarity = np.mean(upper_triangle[upper_triangle > 0])
        
        return {
            'languages': unique_langs,
            'centroid_similarity_matrix': similarity_matrix,
            'average_cross_lingual_similarity': float(avg_similarity),
            'max_similarity': float(np.max(upper_triangle[upper_triangle < 1])),
            'min_similarity': float(np.min(upper_triangle[upper_triangle > 0]))
        }
    
    def analyze_language_clusters(self, features: np.ndarray, languages: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """
        分析语言聚类情况
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            n_clusters: 聚类数量
            
        Returns:
            聚类分析结果
        """
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # 创建语言到聚类的映射
        unique_langs = sorted(set(languages))
        lang_to_cluster = {}
        for lang in unique_langs:
            indices = [i for i, l in enumerate(languages) if l == lang]
            # 使用该语言样本的聚类标签众数
            clusters = [cluster_labels[i] for i in indices]
            lang_to_cluster[lang] = max(set(clusters), key=clusters.count)
        
        # 计算真实语言标签
        lang_to_idx = {lang: idx for idx, lang in enumerate(unique_langs)}
        true_labels = [lang_to_idx[lang] for lang in languages]
        
        # 计算聚类评估指标
        try:
            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            sil_score = silhouette_score(features, cluster_labels)
        except:
            ari_score = 0.0
            sil_score = 0.0
        
        return {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'language_to_cluster': lang_to_cluster,
            'adjusted_rand_index': float(ari_score),
            'silhouette_score': float(sil_score),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
    
    def compare_intra_vs_inter_language_distances(self, features: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """
        比较语言内部和语言间的距离分布
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            
        Returns:
            距离比较结果
        """
        unique_langs = sorted(set(languages))
        intra_distances = []
        inter_distances = []
        
        # 计算语言内部距离
        for lang in unique_langs:
            indices = [i for i, l in enumerate(languages) if l == lang]
            if len(indices) > 1:
                lang_features = features[indices]
                distances = euclidean_distances(lang_features, lang_features)
                # 获取上三角矩阵（不包括对角线）
                mask = np.triu(np.ones(distances.shape, dtype=bool), k=1)
                intra_distances.extend(distances[mask])
        
        # 计算语言间距离
        for i, lang1 in enumerate(unique_langs):
            for j, lang2 in enumerate(unique_langs):
                if i < j:  # 避免重复计算
                    indices1 = [idx for idx, l in enumerate(languages) if l == lang1]
                    indices2 = [idx for idx, l in enumerate(languages) if l == lang2]
                    
                    if len(indices1) > 0 and len(indices2) > 0:
                        features1 = features[indices1]
                        features2 = features[indices2]
                        distances = euclidean_distances(features1, features2)
                        inter_distances.extend(distances.flatten())
        
        # 转换为numpy数组并处理空数组
        intra_distances = np.array(intra_distances) if intra_distances else np.array([])
        inter_distances = np.array(inter_distances) if inter_distances else np.array([])
        
        # 安全的统计计算函数
        def safe_mean(arr):
            return float(np.nanmean(arr)) if len(arr) > 0 else 0.0
            
        def safe_std(arr):
            return float(np.nanstd(arr)) if len(arr) > 1 else 0.0
            
        def safe_median(arr):
            return float(np.nanmedian(arr)) if len(arr) > 0 else 0.0
            
        def safe_effect_size(inter_dist, intra_dist):
            try:
                if len(inter_dist) > 0 and len(intra_dist) > 0:
                    mean_diff = np.nanmean(inter_dist) - np.nanmean(intra_dist)
                    pooled_var = (np.nanvar(inter_dist) + np.nanvar(intra_dist)) / 2
                    if pooled_var > 0:
                        return float(mean_diff / np.sqrt(pooled_var))
                return 0.0
            except:
                return 0.0
        
        # 执行统计检验
        try:
            if len(intra_distances) > 1 and len(inter_distances) > 1:
                with np.errstate(all='ignore'):  # 忽略数值警告
                    statistic, p_value = mannwhitneyu(intra_distances, inter_distances, alternative='two-sided')
                    significant = bool(p_value < 0.05)  # 确保是Python布尔值
            else:
                statistic, p_value, significant = np.nan, np.nan, False
                logger.warning("Sample size too small for statistical testing")
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            statistic, p_value, significant = np.nan, np.nan, False

        results = {
            'intra_language_distances': {
                'mean': safe_mean(intra_distances),
                'std': safe_std(intra_distances),
                'median': safe_median(intra_distances),
                'count': len(intra_distances)
            },
            'inter_language_distances': {
                'mean': safe_mean(inter_distances),
                'std': safe_std(inter_distances),
                'median': safe_median(inter_distances),
                'count': len(inter_distances)
            },
            'statistical_test': {
                'test': 'Mann-Whitney U',
                'statistic': float(statistic) if not np.isnan(statistic) else None,
                'p_value': float(p_value) if not np.isnan(p_value) else None,
                'significant': significant
            },
            'effect_size': safe_effect_size(inter_distances, intra_distances)
        }
        
        return results
    
    def generate_analysis_report(self, features: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """
        生成完整的分析报告
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            
        Returns:
            完整分析报告
        """
        logger.info("Generating comprehensive analysis report...")
        
        report = {
            'dataset_info': {
                'total_samples': len(languages),
                'feature_dimension': features.shape[1],
                'languages': sorted(set(languages)),
                'samples_per_language': {lang: languages.count(lang) for lang in sorted(set(languages))}
            }
        }
        
        # 跨语言相似性分析
        logger.info("Computing cross-lingual similarity...")
        report['cross_lingual_similarity'] = self.compute_cross_lingual_similarity(features, languages)
        
        # 聚类分析
        logger.info("Performing cluster analysis...")
        for n_clusters in [2, 3, 4]:
            report[f'clustering_{n_clusters}'] = self.analyze_language_clusters(features, languages, n_clusters)
        
        # 语言内外距离比较
        logger.info("Comparing intra vs inter-language distances...")
        report['distance_comparison'] = self.compare_intra_vs_inter_language_distances(features, languages)
        
        logger.info("Analysis report generated successfully")
        return report
    
    def save_analysis_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        保存分析报告到JSON文件
        
        Args:
            report: 分析报告字典
            output_path: 输出文件路径
        """
        import json
        import warnings
        
        # 转换numpy数组为列表以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):  # 添加布尔类型处理
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif hasattr(obj, 'item'):  # 处理其他numpy标量类型
                try:
                    return obj.item()
                except (ValueError, TypeError):
                    return str(obj)
            elif isinstance(obj, (int, float, complex)) and np.isnan(obj):
                return None  # 将NaN转换为null
            else:
                return obj
        
        # 抑制警告并处理特殊值
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            serializable_report = convert_numpy(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis report saved to {output_path}")