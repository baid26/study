import numpy as np
import joblib
import torch
import os
import hashlib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from .config import Config

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def init_bert_model(self):
        """初始化BERT模型和分词器"""
        if self.bert_model is not None:
            return
            
        print(f"加载BERT模型: {Config.BERT_MODEL_NAME}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert_model.eval()
        
        if torch.cuda.is_available():
            print("使用GPU加速BERT特征提取")
            self.bert_model = self.bert_model.to('cuda')
        else:
            print("警告: 未检测到GPU，使用CPU模式")
    
    def fit_tfidf(self, texts):
        """训练TF-IDF向量化器"""
        print(f"训练TF-IDF向量化器 (max_features={Config.TFIDF_MAX_FEATURES}, ngram_range={Config.TFIDF_NGRAM_RANGE})")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words='english'
        )
        return self.tfidf_vectorizer.fit_transform(texts)
    
    def transform_tfidf(self, texts):
        """应用TF-IDF转换"""
        return self.tfidf_vectorizer.transform(texts)
    
    def _get_cache_key(self, texts, feature_type):
        """生成缓存键"""
        # 使用文本内容的哈希值作为缓存键
        text_content = "".join(texts)
        hash_key = hashlib.md5(text_content.encode()).hexdigest()
        return f"{feature_type}_{hash_key}_{Config.BERT_MODEL_NAME}_{Config.MAX_SEQ_LENGTH}.npy"
    
    def _load_from_cache(self, cache_key):
        """从缓存加载特征"""
        cache_path = os.path.join(self.cache_dir, cache_key)
        if os.path.exists(cache_path):
            print(f"从缓存加载特征: {cache_path}")
            return np.load(cache_path)
        return None
    
    def _save_to_cache(self, features, cache_key):
        """保存特征到缓存"""
        cache_path = os.path.join(self.cache_dir, cache_key)
        np.save(cache_path, features)
        print(f"特征已缓存: {cache_path}")
    
    def extract_bert_features(self, texts, batch_size=32, use_cache=True):
        """提取BERT特征（带缓存）"""
        if use_cache:
            cache_key = self._get_cache_key(texts, "bert")
            cached_features = self._load_from_cache(cache_key)
            if cached_features is not None:
                return cached_features
        
        if not self.bert_model:
            self.init_bert_model()
            
        features = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 使用进度条
        for i in tqdm(range(0, len(texts), batch_size), desc="提取BERT特征"):
            batch = texts[i:i+batch_size]
            
            # 分词
            inputs = self.bert_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=Config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
            
            # 将输入数据移动到与模型相同的设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 特征提取
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 使用平均池化作为文本表示
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                mean_pooling = torch.sum(last_hidden_state * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                features.append(mean_pooling.cpu().numpy())
        
        features = np.vstack(features)
        
        if use_cache:
            self._save_to_cache(features, cache_key)
            
        return features
    
    def extract_features(self, texts, fit=False, use_cache=True):
        """提取组合特征（带缓存）"""
        # 生成组合特征缓存键
        combined_cache_key = self._get_cache_key(texts, "combined")
        if use_cache:
            cached_features = self._load_from_cache(combined_cache_key)
            if cached_features is not None:
                return cached_features
        
        # TF-IDF特征
        if fit:
            # 如果fit=True，必须重新训练TF-IDF
            print("训练TF-IDF向量化器...")
            tfidf_features = self.fit_tfidf(texts)
            # 保存TF-IDF向量化器
            joblib.dump(self.tfidf_vectorizer, Config.TFIDF_SAVE_PATH)
            print(f"TF-IDF向量化器已保存到: {Config.TFIDF_SAVE_PATH}")
        else:
            # 加载TF-IDF向量化器
            if self.tfidf_vectorizer is None:
                print(f"加载TF-IDF向量化器: {Config.TFIDF_SAVE_PATH}")
                self.tfidf_vectorizer = joblib.load(Config.TFIDF_SAVE_PATH)
            tfidf_features = self.transform_tfidf(texts)
        
        # BERT特征
        bert_features = self.extract_bert_features(texts, use_cache=use_cache)
        
        # 组合特征
        print(f"组合特征: TF-IDF维度={tfidf_features.shape[1]}, BERT维度={bert_features.shape[1]}")
        combined_features = np.hstack([tfidf_features.toarray(), bert_features])
        print(f"最终特征维度: {combined_features.shape[1]}")
        
        if use_cache:
            self._save_to_cache(combined_features, combined_cache_key)
            
        return combined_features
    
    def save(self, tfidf_path, model_dir=None):
        """保存特征提取器"""
        joblib.dump(self.tfidf_vectorizer, tfidf_path)
        if model_dir:
            self.bert_model.save_pretrained(model_dir)
            self.bert_tokenizer.save_pretrained(model_dir)
    
    @classmethod
    def load(cls, tfidf_path, model_dir):
        """加载特征提取器"""
        extractor = cls()
        extractor.tfidf_vectorizer = joblib.load(tfidf_path)
        extractor.bert_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        extractor.bert_model = AutoModel.from_pretrained(model_dir)
        extractor.bert_model.eval()
        return extractor
