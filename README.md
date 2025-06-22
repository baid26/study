# study
大模型生成文本检测
project/
│
├── data/                   # 数据目录
│   ├── train.jsonl         # 训练数据文件，每行为一个JSON对象，包含"text"和"label"字段
│   └── test.jsonl          # 测试数据文件，每行为一个JSON对象，包含"text"字段（无标签）
│
├── models/                 # 模型保存目录
│   ├── xgboost_model.json   # 训练好的XGBoost模型
│   └── tfidf_vectorizer.pkl # TF-IDF向量化器
│
├── features_cache/         # 特征缓存目录（自动创建）
│   ├── bert_*.npy          # BERT特征缓存文件（*为哈希值）
│   └── combined_*.npy      # 组合特征缓存文件（*为哈希值）
│
├── results/                # 结果输出目录
│   └── submit.txt     # 测试集的预测结果，每行一个预测标签（0或1）
│
├── src/                    # 源代码目录
│   ├── config.py           # 配置文件：定义路径、超参数等
│   ├── data_loader.py      # 数据加载模块：读取和处理数据
│   ├── feature_extractor.py # 特征工程模块：提取TF-IDF和BERT特征
│   ├── model_trainer.py    # 模型训练模块：训练和评估XGBoost模型
│   └── predictor.py        # 预测模块：保存预测结果
│
└── main.py                 # 主程序入口：执行整个流程
