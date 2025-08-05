import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import json
import base64
from typing import Dict, List, Optional

from config import Config
from secret_sharing import secret_sharing, secure_comm
from model_utils import embedding_model

# 配置日志
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="本地计算服务器",
    description="基于秘密分享协议的分布式大模型推理引擎 - 本地计算节点",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class EmbeddingRequest(BaseModel):
    share_data: str  # base64编码的分享数据
    x_coord: int
    request_id: str

class EmbeddingResponse(BaseModel):
    result: str  # base64编码的计算结果
    request_id: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str = "healthy"
    server_type: str = "local"
    model_loaded: bool = True

# 全局变量
active_requests: Dict[str, Dict] = {}

@app.get("/", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        server_type="local",
        model_loaded=embedding_model.model is not None
    )

@app.post("/api/v1/compute_embedding", response_model=EmbeddingResponse)
async def compute_embedding(request: EmbeddingRequest):
    """
    计算本地embedding分享
    
    Args:
        request: 包含分享数据和坐标的请求
        
    Returns:
        计算结果的分享
    """
    try:
        logger.info(f"收到embedding计算请求: {request.request_id}")
        
        # 解码分享数据
        share_bytes = base64.b64decode(request.share_data.encode('utf-8'))
        share_array = np.frombuffer(share_bytes, dtype=np.float32)
        
        # 重塑为原始形状（假设是1维embedding）
        share_reshaped = share_array.reshape(-1)
        
        logger.info(f"分享数据形状: {share_reshaped.shape}")
        
        # 使用本地模型计算embedding
        # 这里我们模拟embedding计算过程
        # 在实际应用中，这里会使用真实的模型计算
        
        # 将分享数据转换为文本（简化处理）
        # 在实际应用中，这里需要更复杂的处理逻辑
        text_representation = f"share_{request.x_coord}"
        
        # 计算embedding
        embedding = embedding_model.compute_embedding(text_representation)
        
        # 将embedding与分享数据结合（模拟计算过程）
        # 在实际应用中，这里需要根据具体的秘密分享协议进行计算
        result_embedding = embedding + share_reshaped.reshape(embedding.shape) * 0.1
        
        # 编码结果
        result_bytes = result_embedding.tobytes()
        result_b64 = base64.b64encode(result_bytes).decode('utf-8')
        
        # 存储请求信息
        active_requests[request.request_id] = {
            "x_coord": request.x_coord,
            "input_shape": share_reshaped.shape,
            "output_shape": result_embedding.shape,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"embedding计算完成: {request.request_id}")
        
        return EmbeddingResponse(
            result=result_b64,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"计算embedding时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def get_status():
    """获取服务器状态"""
    return {
        "server_type": "local",
        "active_requests": len(active_requests),
        "model_loaded": embedding_model.model is not None,
        "model_name": embedding_model.model_name,
        "device": str(embedding_model.device)
    }

@app.get("/api/v1/requests/{request_id}")
async def get_request_info(request_id: str):
    """获取特定请求的信息"""
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="请求未找到")
    
    return active_requests[request_id]

@app.delete("/api/v1/requests/{request_id}")
async def clear_request(request_id: str):
    """清除特定请求的信息"""
    if request_id in active_requests:
        del active_requests[request_id]
        return {"status": "cleared"}
    else:
        raise HTTPException(status_code=404, detail="请求未找到")

@app.post("/api/v1/test_embedding")
async def test_embedding():
    """测试embedding计算功能"""
    try:
        test_text = "这是一个测试文本"
        embedding = embedding_model.compute_embedding(test_text)
        
        return {
            "status": "success",
            "test_text": test_text,
            "embedding_shape": embedding.shape,
            "embedding_sample": embedding.flatten()[:5].tolist()
        }
    except Exception as e:
        logger.error(f"测试embedding时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("本地计算服务器启动中...")
    Config.create_dirs()
    logger.info(f"服务器将在 http://{Config.LOCAL_HOST}:{Config.LOCAL_PORT} 启动")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("本地计算服务器关闭中...")

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "local_server:app",
        host=Config.LOCAL_HOST,
        port=Config.LOCAL_PORT,
        reload=True,
        log_level=Config.LOG_LEVEL.lower()
    ) 