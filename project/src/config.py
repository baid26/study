import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # src的父目录是项目根目录

class Config:
    # 使用基于项目根目录的路径
    DATA_DIR = os.path.join(project_root, "data/")
    TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
    TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
    MODEL_SAVE_PATH = os.path.join(project_root, "models", "xgboost_model.json")
    TFIDF_SAVE_PATH = os.path.join(project_root, "models", "tfidf_vectorizer.pkl")
    RESULTS_DIR = os.path.join(project_root, "results/")
    CACHE_DIR = os.path.join(project_root, "features_cache/")  # 新增缓存目录
    
    # 特征工程配置
    TFIDF_MAX_FEATURES = 10000
    TFIDF_NGRAM_RANGE = (1, 3)
    BERT_MODEL_NAME = "distilbert-base-uncased"
    MAX_SEQ_LENGTH = 256
    BERT_BATCH_SIZE = 32
    
    # XGBoost参数
    XGB_PARAMS = {
        'n_estimators': 800,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    # 训练参数
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EARLY_STOPPING_ROUNDS = 30
