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
        centroids = self.compute_language_centroids(features, languages)
        unique_langs = sorted(centroids.keys())
        
        # 构建中心向量矩阵
        centroid_matrix = np.array([centroids[lang] for lang in unique_langs])
        
        # 计算相似性矩阵
        similarity_matrix = cosine_similarity(centroid_matrix)
        distance_matrix = euclidean_distances(centroid_matrix)
        
        # 计算平均相似性和距离
        n_langs = len(unique_langs)
        upper_triangle_indices = np.triu_indices(n_langs, k=1)
        
        similarities = similarity_matrix[upper_triangle_indices]
        distances = distance_matrix[upper_triangle_indices]
        
        results = {
            'language_pairs': [(unique_langs[i], unique_langs[j]) 
                             for i, j in zip(*upper_triangle_indices)],
            'similarity_matrix': similarity_matrix,
            'distance_matrix': distance_matrix,
            'language_names': unique_langs,
            'avg_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'avg_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances))
        }
        
        # 找出最相似和最不相似的语言对
        min_sim_idx = np.argmin(similarities)
        max_sim_idx = np.argmax(similarities)
        
        results['most_similar_pair'] = results['language_pairs'][max_sim_idx]
        results['least_similar_pair'] = results['language_pairs'][min_sim_idx]
        results['most_similar_score'] = float(similarities[max_sim_idx])
        results['least_similar_score'] = float(similarities[min_sim_idx])
        
        return results
    
    def analyze_language_clusters(self, features: np.ndarray, languages: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """
        分析语言聚类模式
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            n_clusters: 聚类数量
            
        Returns:
            聚类分析结果
        """
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        cluster_labels = kmeans.fit_predict(features)
        
        # 分析每个聚类的语言分布
        cluster_analysis = {}
        unique_langs = sorted(set(languages))
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_languages = [languages[i] for i in cluster_indices]
            
            # 统计每种语言在该聚类中的数量
            lang_counts = {}
            for lang in cluster_languages:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_indices),
                'language_distribution': lang_counts,
                'dominant_language': max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else None
            }
        
        # 计算语言混合度（衡量不同语言在聚类中的分布均匀性）
        mixing_scores = []
        for cluster_id in range(n_clusters):
            cluster_info = cluster_analysis[f'cluster_{cluster_id}']
            lang_dist = cluster_info['language_distribution']
            if len(lang_dist) > 1:
                # 计算熵作为混合度指标
                total = sum(lang_dist.values())
                probs = [count/total for count in lang_dist.values()]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                mixing_scores.append(entropy)
            else:
                mixing_scores.append(0.0)
        
        results = {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_analysis': cluster_analysis,
            'n_clusters': n_clusters,
            'mixing_scores': mixing_scores,
            'avg_mixing_score': float(np.mean(mixing_scores)),
            'silhouette_score': self._compute_silhouette_score(features, cluster_labels)
        }
        
        return results
    
    def _compute_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """计算轮廓系数"""
        from sklearn.metrics import silhouette_score
        try:
            return float(silhouette_score(features, labels))
        except:
            return 0.0
    
    def compare_intra_vs_inter_language_distances(self, features: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """
        比较语言内距离和语言间距离
        
        Args:
            features: 特征矩阵
            languages: 语言标签列表
            
        Returns:
            距离比较分析结果
        """
        unique_langs = sorted(set(languages))
        intra_distances = []
        inter_distances = []
        
        # 计算语言内距离
        for lang in unique_langs:
            indices = [i for i, l in enumerate(languages) if l == lang]
            if len(indices) > 1:
                lang_features = features[indices]
                # 计算该语言内所有样本对的距离
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        dist = euclidean_distances([lang_features[i]], [lang_features[j]])[0][0]
                        intra_distances.append(dist)
        
        # 计算语言间距离
        for i, lang1 in enumerate(unique_langs):
            for j, lang2 in enumerate(unique_langs):
                if i < j:  # 避免重复计算
                    indices1 = [idx for idx, l in enumerate(languages) if l == lang1]
                    indices2 = [idx for idx, l in enumerate(languages) if l == lang2]
                    
                    # 计算两种语言间所有样本对的距离
                    for idx1 in indices1:
                        for idx2 in indices2:
                            dist = euclidean_distances([features[idx1]], [features[idx2]])[0][0]
                            inter_distances.append(dist)
        
        # 统计分析
        intra_distances = np.array(intra_distances)
        inter_distances = np.array(inter_distances)
        
        # 执行统计检验
        statistic, p_value = mannwhitneyu(intra_distances, inter_distances, alternative='two-sided')
        
        results = {
            'intra_language_distances': {
                'mean': float(np.mean(intra_distances)),
                'std': float(np.std(intra_distances)),
                'median': float(np.median(intra_distances)),
                'count': len(intra_distances)
            },
            'inter_language_distances': {
                'mean': float(np.mean(inter_distances)),
                'std': float(np.std(inter_distances)),
                'median': float(np.median(inter_distances)),
                'count': len(inter_distances)
            },
            'statistical_test': {
                'test': 'Mann-Whitney U',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'effect_size': float((np.mean(inter_distances) - np.mean(intra_distances)) / 
                               np.sqrt((np.var(inter_distances) + np.var(intra_distances)) / 2))
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
        
        # 转换numpy数组为列表以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_report = convert_numpy(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis report saved to {output_path}")