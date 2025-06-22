# study
大模型生成文本检测
# 项目文件结构树

```plaintext
project/
├── data/                    # 数据存储目录
│   ├── train.jsonl
│   └── test.jsonl
├── models/                  # 模型存储目录
│   ├── xgboost_model.json
│   └── tfidf_vectorizer.pkl
├── features_cache/          # 特征缓存目录 (自动生成)
│   ├── bert_*.npy
│   └── combined_*.npy
├── results/                 # 结果输出目录
│   └── submit.txt
├── src/                     # 源代码目录
│   ├── config.py            # 配置文件：路径和超参数管理
│   ├── data_loader.py       # 数据加载模块：读取和处理数据
│   ├── feature_extractor.py # 特征工程模块：提取TF-IDF和BERT特征
│   ├── model_trainer.py     # 模型训练模块：训练和评估XGBoost
│   └── predictor.py         # 预测模块：保存预测结果
└── main.py                  # 主程序入口：协调整个流程
