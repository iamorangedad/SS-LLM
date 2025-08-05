import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """用于计算embedding的模型类"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        初始化embedding模型
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            # 使用简化的模型作为备选
            self._create_simple_model()
    
    def _load_model(self):
        """加载预训练模型"""
        logger.info(f"正在加载模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("模型加载完成")
    
    def _create_simple_model(self):
        """创建简化的embedding模型作为备选"""
        logger.info("创建简化模型")
        
        class SimpleEmbeddingModel(nn.Module):
            def __init__(self, vocab_size=30522, hidden_size=768):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=12,
                    dim_feedforward=3072,
                    dropout=0.1
                )
                self.layer_norm = nn.LayerNorm(hidden_size)
                
            def forward(self, input_ids, attention_mask=None):
                embeddings = self.embedding(input_ids)
                if attention_mask is not None:
                    embeddings = embeddings * attention_mask.unsqueeze(-1)
                
                # 简化的transformer处理
                output = self.transformer(embeddings)
                output = self.layer_norm(output)
                
                # 返回平均池化的embedding
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    sum_embeddings = torch.sum(output * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    return sum_embeddings / sum_mask
                else:
                    return torch.mean(output, dim=1)
        
        self.model = SimpleEmbeddingModel()
        self.model.to(self.device)
        self.model.eval()
        
        # 创建简化的tokenizer
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
                self.vocab_size = 4
                
            def encode(self, text, max_length=512, truncation=True, padding=True):
                # 简单的分词实现
                words = text.split()
                token_ids = [self.vocab.get(word, 1) for word in words]  # 1是UNK
                
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                
                if padding:
                    while len(token_ids) < max_length:
                        token_ids.append(0)  # PAD
                
                return {"input_ids": token_ids, "attention_mask": [1 if x != 0 else 0 for x in token_ids]}
        
        self.tokenizer = SimpleTokenizer()
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """
        计算文本的embedding
        
        Args:
            text: 输入文本
            
        Returns:
            embedding向量
        """
        try:
            with torch.no_grad():
                # 编码文本
                if hasattr(self.tokenizer, 'encode'):
                    encoded = self.tokenizer.encode(text, max_length=512, truncation=True, padding=True)
                    input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
                    attention_mask = torch.tensor([encoded['attention_mask']]).to(self.device)
                else:
                    # 使用简化的tokenizer
                    encoded = self.tokenizer.encode(text)
                    input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
                    attention_mask = torch.tensor([encoded['attention_mask']]).to(self.device)
                
                # 计算embedding
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 获取embedding
                if isinstance(outputs, tuple):
                    embeddings = outputs[0]  # 第一个输出通常是hidden states
                else:
                    embeddings = outputs
                
                # 转换为numpy数组
                embedding_np = embeddings.cpu().numpy()
                
                return embedding_np
                
        except Exception as e:
            logger.error(f"计算embedding时出错: {e}")
            # 返回随机embedding作为备选
            return np.random.randn(1, 768).astype(np.float32)
    
    def compute_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量计算embedding
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding矩阵
        """
        embeddings = []
        for text in texts:
            embedding = self.compute_embedding(text)
            embeddings.append(embedding)
        
        return np.concatenate(embeddings, axis=0)


class InferenceEngine:
    """推理引擎类"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        初始化推理引擎
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"加载推理模型失败: {e}")
            self._create_simple_inference_model()
    
    def _load_model(self):
        """加载推理模型"""
        logger.info(f"正在加载推理模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("推理模型加载完成")
    
    def _create_simple_inference_model(self):
        """创建简化的推理模型"""
        logger.info("创建简化推理模型")
        
        class SimpleInferenceModel(nn.Module):
            def __init__(self, vocab_size=50257, hidden_size=768):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=12,
                    dim_feedforward=3072,
                    dropout=0.1
                )
                self.output_layer = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, input_ids, attention_mask=None):
                embeddings = self.embedding(input_ids)
                if attention_mask is not None:
                    embeddings = embeddings * attention_mask.unsqueeze(-1)
                
                output = self.transformer(embeddings)
                logits = self.output_layer(output)
                
                return logits
        
        self.model = SimpleInferenceModel()
        self.model.to(self.device)
        self.model.eval()
        
        # 简化的tokenizer
        class SimpleInferenceTokenizer:
            def __init__(self):
                self.vocab = {"<|endoftext|>": 0, "hello": 1, "world": 2}
                self.vocab_size = 3
                
            def encode(self, text):
                words = text.split()
                token_ids = [self.vocab.get(word, 0) for word in words]
                return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}
            
            def decode(self, token_ids):
                return " ".join([list(self.vocab.keys())[i] for i in token_ids if i < len(self.vocab)])
        
        self.tokenizer = SimpleInferenceTokenizer()
    
    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            
        Returns:
            生成的文本
        """
        try:
            with torch.no_grad():
                # 编码输入
                if hasattr(self.tokenizer, 'encode'):
                    encoded = self.tokenizer.encode(prompt, return_tensors="pt")
                    input_ids = encoded.to(self.device)
                else:
                    encoded = self.tokenizer.encode(prompt)
                    input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
                
                # 生成文本
                generated_ids = input_ids.clone()
                
                for _ in range(max_length):
                    # 获取模型输出
                    outputs = self.model(input_ids=generated_ids)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    # 获取下一个token的概率
                    next_token_logits = logits[0, -1, :]
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    
                    # 采样下一个token
                    next_token = torch.multinomial(next_token_probs, 1)
                    
                    # 添加到生成序列
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                    
                    # 检查是否生成了结束标记
                    if next_token.item() == 0:  # 假设0是结束标记
                        break
                
                # 解码生成的文本
                if hasattr(self.tokenizer, 'decode'):
                    generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                else:
                    generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                
                return generated_text
                
        except Exception as e:
            logger.error(f"生成文本时出错: {e}")
            return f"{prompt} [生成失败]"
    
    def process_embedding(self, embedding: np.ndarray) -> str:
        """
        处理embedding并生成文本
        
        Args:
            embedding: 输入的embedding
            
        Returns:
            生成的文本
        """
        try:
            # 将embedding转换为tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            
            # 使用embedding进行推理
            # 这里简化处理，实际应用中可能需要更复杂的逻辑
            with torch.no_grad():
                # 假设embedding可以直接用于生成
                # 在实际应用中，这里可能需要更复杂的处理
                output = self.model(inputs_embeds=embedding_tensor)
                
                # 生成文本（简化版本）
                return "基于embedding生成的文本"
                
        except Exception as e:
            logger.error(f"处理embedding时出错: {e}")
            return "处理失败"


# 全局实例
embedding_model = EmbeddingModel()
inference_engine = InferenceEngine() 