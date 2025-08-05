import numpy as np
import random
from typing import Tuple, List, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import json


class SecretSharing:
    """基于Shamir秘密分享协议的实现"""
    
    def __init__(self, prime: int = 2**31 - 1):
        """
        初始化秘密分享协议
        
        Args:
            prime: 用于有限域运算的素数
        """
        self.prime = prime
        self.field_size = prime
    
    def split_secret(self, secret: np.ndarray, n_shares: int = 2) -> Tuple[List[np.ndarray], List[int]]:
        """
        将秘密数据拆分成n份分享
        
        Args:
            secret: 要分享的秘密数据
            n_shares: 分享数量
            
        Returns:
            shares: 分享数据列表
            x_coords: x坐标列表
        """
        if n_shares < 2:
            raise ValueError("分享数量必须至少为2")
        
        # 生成随机系数
        coefficients = []
        for i in range(secret.shape[0]):
            row_coeffs = [secret[i]]  # 常数项是秘密值
            for _ in range(n_shares - 1):
                row_coeffs.append(random.randint(0, self.prime - 1))
            coefficients.append(row_coeffs)
        
        # 生成x坐标
        x_coords = [i + 1 for i in range(n_shares)]
        
        # 计算分享值
        shares = []
        for x in x_coords:
            share = np.zeros_like(secret)
            for i in range(secret.shape[0]):
                # 计算多项式在x处的值
                value = 0
                for j, coeff in enumerate(coefficients[i]):
                    value = (value + coeff * (x ** j)) % self.prime
                share[i] = value
            shares.append(share)
        
        return shares, x_coords
    
    def reconstruct_secret(self, shares: List[np.ndarray], x_coords: List[int]) -> np.ndarray:
        """
        从分享中重构原始秘密
        
        Args:
            shares: 分享数据列表
            x_coords: x坐标列表
            
        Returns:
            重构的秘密数据
        """
        if len(shares) != len(x_coords):
            raise ValueError("分享数量和x坐标数量必须相同")
        
        if len(shares) < 2:
            raise ValueError("至少需要2个分享才能重构")
        
        # 使用拉格朗日插值重构秘密
        secret = np.zeros_like(shares[0])
        
        for i in range(secret.shape[0]):
            # 对每个维度使用拉格朗日插值
            value = 0
            for j, (share, x_j) in enumerate(zip(shares, x_coords)):
                # 计算拉格朗日基多项式
                numerator = denominator = 1
                for k, x_k in enumerate(x_coords):
                    if k != j:
                        numerator = (numerator * (-x_k)) % self.prime
                        denominator = (denominator * (x_j - x_k)) % self.prime
                
                # 计算拉格朗日系数
                if denominator == 0:
                    raise ValueError("x坐标不能重复")
                
                # 计算模逆元
                inv_denominator = pow(denominator, -1, self.prime)
                lagrange_coeff = (numerator * inv_denominator) % self.prime
                
                # 累加到结果
                value = (value + share[i] * lagrange_coeff) % self.prime
            
            secret[i] = value
        
        return secret
    
    def split_text(self, text: str, n_shares: int = 2) -> Tuple[List[str], List[int]]:
        """
        将文本数据拆分成分享
        
        Args:
            text: 要分享的文本
            n_shares: 分享数量
            
        Returns:
            shares: 分享文本列表
            x_coords: x坐标列表
        """
        # 将文本转换为字节数组
        text_bytes = text.encode('utf-8')
        text_array = np.array([b for b in text_bytes], dtype=np.int32)
        
        # 拆分秘密
        shares, x_coords = self.split_secret(text_array, n_shares)
        
        # 将分享转换为base64编码的字符串
        share_strings = []
        for share in shares:
            share_bytes = bytes([int(x) % 256 for x in share])
            share_string = base64.b64encode(share_bytes).decode('utf-8')
            share_strings.append(share_string)
        
        return share_strings, x_coords
    
    def reconstruct_text(self, shares: List[str], x_coords: List[int]) -> str:
        """
        从分享中重构原始文本
        
        Args:
            shares: 分享文本列表
            x_coords: x坐标列表
            
        Returns:
            重构的原始文本
        """
        # 将base64编码的分享转换回数组
        share_arrays = []
        for share_str in shares:
            share_bytes = base64.b64decode(share_str.encode('utf-8'))
            share_array = np.array([b for b in share_bytes], dtype=np.int32)
            share_arrays.append(share_array)
        
        # 重构秘密
        secret_array = self.reconstruct_secret(share_arrays, x_coords)
        
        # 转换回文本
        secret_bytes = bytes([int(x) % 256 for x in secret_array])
        return secret_bytes.decode('utf-8')
    
    def split_embedding(self, embedding: np.ndarray, n_shares: int = 2) -> Tuple[List[np.ndarray], List[int]]:
        """
        将embedding向量拆分成分享
        
        Args:
            embedding: embedding向量
            n_shares: 分享数量
            
        Returns:
            shares: 分享embedding列表
            x_coords: x坐标列表
        """
        # 将embedding展平
        flat_embedding = embedding.flatten()
        
        # 拆分秘密
        shares, x_coords = self.split_secret(flat_embedding, n_shares)
        
        # 将分享重塑为原始形状
        shaped_shares = []
        for share in shares:
            shaped_share = share.reshape(embedding.shape)
            shaped_shares.append(shaped_share)
        
        return shaped_shares, x_coords
    
    def reconstruct_embedding(self, shares: List[np.ndarray], x_coords: List[int]) -> np.ndarray:
        """
        从分享中重构原始embedding
        
        Args:
            shares: 分享embedding列表
            x_coords: x坐标列表
            
        Returns:
            重构的原始embedding
        """
        # 将embedding展平
        flat_shares = [share.flatten() for share in shares]
        
        # 重构秘密
        secret_flat = self.reconstruct_secret(flat_shares, x_coords)
        
        # 重塑为原始形状
        return secret_flat.reshape(shares[0].shape)


class SecureCommunication:
    """安全通信工具类"""
    
    def __init__(self, key: Optional[bytes] = None):
        """
        初始化安全通信
        
        Args:
            key: 加密密钥，如果为None则生成新密钥
        """
        if key is None:
            key = self._generate_key()
        self.key = key
    
    def _generate_key(self) -> bytes:
        """生成随机密钥"""
        return random.getrandbits(256).to_bytes(32, 'big')
    
    def encrypt_data(self, data: dict) -> str:
        """
        加密数据
        
        Args:
            data: 要加密的数据字典
            
        Returns:
            加密后的base64字符串
        """
        # 将数据转换为JSON字符串
        json_str = json.dumps(data, default=str)
        
        # 使用密钥生成加密密钥
        salt = b'salt_123'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(self.key)
        
        # 简单的XOR加密（实际应用中应使用更安全的加密）
        encrypted = bytes(a ^ b for a, b in zip(json_str.encode(), key))
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> dict:
        """
        解密数据
        
        Args:
            encrypted_data: 加密的base64字符串
            
        Returns:
            解密后的数据字典
        """
        # 解码base64
        encrypted = base64.b64decode(encrypted_data.encode('utf-8'))
        
        # 使用密钥生成解密密钥
        salt = b'salt_123'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(self.key)
        
        # 简单的XOR解密
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
        
        return json.loads(decrypted.decode('utf-8'))


# 全局实例
secret_sharing = SecretSharing()
secure_comm = SecureCommunication() 