import os
import time
import argparse
import shutil
from src import data_loader, model_trainer, predictor
from src.feature_extractor import FeatureExtractor
from src.config import Config

def main(use_cache=True):
    start_time = time.time()
    
    # 确保缓存目录存在
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    # 步骤1: 加载数据
    print("\n" + "="*50)
    print("步骤1: 加载数据")
    print("="*50)
    X_train, X_val, y_train, y_val, test_texts = data_loader.prepare_datasets()
    
    # 步骤2: 特征工程
    print("\n" + "="*50)
    print("步骤2: 特征工程")
    print(f"使用缓存: {'是' if use_cache else '否'}")
    print("="*50)
    feat_extractor = FeatureExtractor()
    
    print("提取训练集特征...")
    X_train_features = feat_extractor.extract_features(X_train, fit=True, use_cache=use_cache)
    
    print("提取验证集特征...")
    X_val_features = feat_extractor.extract_features(X_val, use_cache=use_cache)
    
    print("提取测试集特征...")
    X_test_features = feat_extractor.extract_features(test_texts, use_cache=use_cache)
    
    # 步骤3: 训练模型
    print("\n" + "="*50)
    print("步骤3: 训练模型")
    print("="*50)
    trainer = model_trainer.ModelTrainer()
    trainer.train(X_train_features, y_train, X_val_features, y_val)
    
    # 保存模型
    trainer.save(Config.MODEL_SAVE_PATH)
    
    # 步骤4: 预测测试集
    print("\n" + "="*50)
    print("步骤4: 预测测试集")
    print("="*50)
    test_preds = trainer.predict(X_test_features)
    
    # 保存结果
    result_path = os.path.join(Config.RESULTS_DIR, "submit.txt")
    predictor.save_predictions(test_preds, result_path)
    
    # 性能统计
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"任务完成! 总耗时: {total_time:.2f}秒")
    print("="*50)

def clean_cache():
    """清理特征缓存"""
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
        print(f"已清理缓存目录: {Config.CACHE_DIR}")
    else:
        print(f"缓存目录不存在: {Config.CACHE_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI文本检测器')
    parser.add_argument('--no-cache', action='store_true', help='禁用特征缓存')
    parser.add_argument('--clean-cache', action='store_true', help='清理特征缓存')
    args = parser.parse_args()
    
    if args.clean_cache:
        clean_cache()
        exit(0)
    
    main(use_cache=not args.no_cache)
