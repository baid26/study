import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from .config import Config

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, y_train, X_val, y_val):
        """训练模型并评估"""
        # 创建 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 参数设置
        params = {
            'objective': 'binary:logistic',
            'max_depth': Config.XGB_PARAMS['max_depth'],
            'learning_rate': Config.XGB_PARAMS['learning_rate'],
            'subsample': Config.XGB_PARAMS['subsample'],
            'colsample_bytree': Config.XGB_PARAMS['colsample_bytree'],
            'eval_metric': 'logloss',
            'seed': Config.XGB_PARAMS['random_state']
        }
        
        # 训练模型
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=Config.XGB_PARAMS['n_estimators'],
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=10
        )
        
        # 验证集评估
        val_preds_proba = self.model.predict(dval)
        val_preds = (val_preds_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        
        print("\n模型评估结果:")
        print(f"验证集准确率: {accuracy:.4f}")
        print(f"验证集F1分数: {f1:.4f}")
        print("\n分类报告:")
        print(classification_report(y_val, val_preds))
        
        return accuracy, f1
    
    def predict(self, X):
        """预测"""
        dtest = xgb.DMatrix(X)
        preds_proba = self.model.predict(dtest)
        return (preds_proba > 0.5).astype(int)
    
    def save(self, path):
        """保存模型"""
        self.model.save_model(path)
    
    @classmethod
    def load(cls, path):
        """加载模型"""
        trainer = cls()
        trainer.model = xgb.Booster()
        trainer.model.load_model(path)
        return trainer
