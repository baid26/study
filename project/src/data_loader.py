import json
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import Config

def load_data(file_path, has_labels=True):
    """加载JSONL格式的数据"""
    texts = []
    labels = [] if has_labels else None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
            if has_labels:
                labels.append(data['label'])
    
    return (texts, labels) if has_labels else texts

def prepare_datasets():
    """准备训练集、验证集和测试集"""
    # 加载训练数据
    train_texts, train_labels = load_data(Config.TRAIN_PATH)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        train_texts, train_labels, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=train_labels  # 保持类别比例
    )
    
    # 加载测试数据
    test_texts = load_data(Config.TEST_PATH, has_labels=False)
    
    print(f"训练样本: {len(X_train)} | 验证样本: {len(X_val)} | 测试样本: {len(test_texts)}")
    return X_train, X_val, y_train, y_val, test_texts
