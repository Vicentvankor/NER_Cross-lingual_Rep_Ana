# NER Cross-lingual Representation Analysis

这是一个用于分析多语言NER模型内部表示的工具，可以帮助研究不同语言之间的相互影响。

## 项目结构

```
.
├── config/
│   └── model_config.py          # 模型和实验配置
├── src/
│   ├── data_loader.py           # 数据加载模块
│   ├── model_manager.py         # 模型管理模块
│   ├── feature_extractor.py     # 特征提取模块
│   ├── visualizer.py            # 可视化模块
│   └── analyzer.py              # 分析模块
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖文件
└── README.md                    # 说明文档
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 基础模型分析（当前可用）

使用未训练的基础模型分析多语言表示：

```bash
# 使用测试数据
python main.py

# 使用真实数据
python main.py --use-real-data

# 自定义参数
python main.py --use-real-data --max-samples 50 --output-dir my_outputs
```

### 参数说明

- `--use-real-data`: 使用Retri_data中的真实数据而非测试样本
- `--base-model`: 指定基础模型路径（默认Qwen/Qwen2.5-7B）
- `--max-samples`: 每种语言的最大样本数（默认100）
- `--output-dir`: 特征输出目录（默认outputs）
- `--results-dir`: 分析结果目录（默认results）

## 功能特性

### 当前功能（仅基础模型）

1. **特征提取**: 从模型隐藏层提取多语言文本表示
2. **可视化分析**: 
   - t-SNE降维可视化
   - PCA降维可视化
   - 语言相似性矩阵热图
   - 语言分布统计图
3. **量化分析**:
   - 跨语言相似性计算
   - 语言聚类分析
   - 语言内外距离对比
   - 统计显著性检验

### 输出文件

运行后会生成以下文件：

```
outputs/
├── base_model_features.npy      # 特征矩阵
├── base_model_languages.txt     # 语言标签
├── base_model_texts.txt         # 原始文本
└── base_model_metadata.txt      # 元数据信息

results/
├── visualizations/
│   ├── base_model_tsne.png
│   ├── base_model_pca.png
│   ├── base_model_similarity_matrix.png
│   └── base_model_language_distribution.png
└── base_model_analysis_report.json
```

## 关于验证跨语言影响的结论

### 当前能验证什么（仅基础模型）

✅ **可以验证的**：
- 基础模型对不同语言的内在表示差异
- 预训练模型是否已具备跨语言理解能力
- 为后续对比实验建立基线

❌ **无法直接验证的**：
- NER微调是否会产生跨语言影响（需要对比微调前后）
- 不同微调策略的影响差异（需要多个微调模型对比）

### 未来扩展（当有LoRA模型时）

当您训练好LoRA模型后，可以扩展以下功能：
1. 多模型对比分析
2. 微调前后表示变化分析
3. 不同微调策略效果对比
4. 跨语言迁移效应量化

## 技术架构

项目采用高内聚低耦合的设计：

- **配置层**: 统一管理模型和实验配置
- **数据层**: 灵活的数据加载和预处理
- **模型层**: 封装模型加载和特征提取
- **分析层**: 模块化的可视化和量化分析
- **接口层**: 清晰的主程序入口和参数管理

每个模块职责单一，便于维护和扩展。