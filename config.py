import os
from typing import Dict, Any

class Config:
    """系统配置类"""
    
    # 服务器配置
    LOCAL_HOST = "localhost"
    LOCAL_PORT = 8001
    CLOUD_HOST = "localhost"
    CLOUD_PORT = 8002
    
    # 模型配置
    EMBEDDING_MODEL = "bert-base-uncased"
    INFERENCE_MODEL = "gpt2"
    
    # 秘密分享配置
    PRIME = 2**31 - 1  # 用于有限域运算的素数
    SHARE_COUNT = 2  # 分享数量
    
    # 安全配置
    ENCRYPTION_KEY_SIZE = 256  # 密钥大小（位）
    SALT = b'salt_123'  # 加密盐值
    
    # 网络配置
    TIMEOUT = 30  # 请求超时时间（秒）
    MAX_RETRIES = 3  # 最大重试次数
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 模型缓存配置
    MODEL_CACHE_DIR = "./model_cache"
    
    # API配置
    API_VERSION = "v1"
    API_PREFIX = f"/api/{API_VERSION}"
    
    @classmethod
    def get_local_url(cls) -> str:
        """获取本地服务器URL"""
        return f"http://{cls.LOCAL_HOST}:{cls.LOCAL_PORT}"
    
    @classmethod
    def get_cloud_url(cls) -> str:
        """获取云端服务器URL"""
        return f"http://{cls.CLOUD_HOST}:{cls.CLOUD_PORT}"
    
    @classmethod
    def get_api_url(cls, server_type: str) -> str:
        """获取API URL"""
        base_url = cls.get_local_url() if server_type == "local" else cls.get_cloud_url()
        return f"{base_url}{cls.API_PREFIX}"
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "inference_model": cls.INFERENCE_MODEL,
            "cache_dir": cls.MODEL_CACHE_DIR
        }
    
    @classmethod
    def get_security_config(cls) -> Dict[str, Any]:
        """获取安全配置"""
        return {
            "prime": cls.PRIME,
            "share_count": cls.SHARE_COUNT,
            "key_size": cls.ENCRYPTION_KEY_SIZE,
            "salt": cls.SALT
        }
    
    @classmethod
    def get_server_config(cls) -> Dict[str, Any]:
        """获取服务器配置"""
        return {
            "local": {
                "host": cls.LOCAL_HOST,
                "port": cls.LOCAL_PORT,
                "url": cls.get_local_url()
            },
            "cloud": {
                "host": cls.CLOUD_HOST,
                "port": cls.CLOUD_PORT,
                "url": cls.get_cloud_url()
            }
        }


# 环境变量覆盖
def load_env_config():
    """从环境变量加载配置"""
    import os
    
    # 服务器配置
    if os.getenv("LOCAL_HOST"):
        Config.LOCAL_HOST = os.getenv("LOCAL_HOST")
    if os.getenv("LOCAL_PORT"):
        Config.LOCAL_PORT = int(os.getenv("LOCAL_PORT"))
    if os.getenv("CLOUD_HOST"):
        Config.CLOUD_HOST = os.getenv("CLOUD_HOST")
    if os.getenv("CLOUD_PORT"):
        Config.CLOUD_PORT = int(os.getenv("CLOUD_PORT"))
    
    # 模型配置
    if os.getenv("EMBEDDING_MODEL"):
        Config.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    if os.getenv("INFERENCE_MODEL"):
        Config.INFERENCE_MODEL = os.getenv("INFERENCE_MODEL")
    
    # 日志配置
    if os.getenv("LOG_LEVEL"):
        Config.LOG_LEVEL = os.getenv("LOG_LEVEL")


# 加载环境配置
load_env_config() 