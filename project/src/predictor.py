import os
import numpy as np
from .config import Config

def save_predictions(predictions, file_path):
    """将预测结果保存为TXT文件"""
    # 确保结果目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存预测结果（每行一个0或1）
    with open(file_path, 'w') as f:
        for pred in predictions:
            f.write(f"{int(pred)}\n")
    
    print(f"预测结果已保存至: {file_path}")
    print(f"总样本数: {len(predictions)}")
    print(f"AI生成文本比例: {np.mean(predictions):.2%}")
