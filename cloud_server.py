import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import json
import base64
import requests
from typing import Dict, List, Optional, Tuple

from config import Config
from secret_sharing import secret_sharing, secure_comm
from model_utils import embedding_model, inference_engine

# 配置日志
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="云端计算服务器",
    description="基于秘密分享协议的分布式大模型推理引擎 - 云端计算节点",
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

class MergeRequest(BaseModel):
    local_result: str  # 本地计算结果
    cloud_result: str  # 云端计算结果
    x_coords: List[int]  # x坐标列表
    request_id: str

class MergeResponse(BaseModel):
    merged_embedding: str  # 合并后的embedding
    final_result: str  # 最终推理结果
    request_id: str
    status: str = "success"

class InferenceRequest(BaseModel):
    text: str
    request_id: str

class InferenceResponse(BaseModel):
    result: str
    request_id: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str = "healthy"
    server_type: str = "cloud"
    model_loaded: bool = True

# 全局变量
active_requests: Dict[str, Dict] = {}
pending_merges: Dict[str, Dict] = {}

@app.get("/", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        server_type="cloud",
        model_loaded=embedding_model.model is not None
    )

@app.post("/api/v1/compute_embedding", response_model=EmbeddingResponse)
async def compute_embedding(request: EmbeddingRequest):
    """
    计算云端embedding分享
    
    Args:
        request: 包含分享数据和坐标的请求
        
    Returns:
        计算结果的分享
    """
    try:
        logger.info(f"收到云端embedding计算请求: {request.request_id}")
        
        # 解码分享数据
        share_bytes = base64.b64decode(request.share_data.encode('utf-8'))
        share_array = np.frombuffer(share_bytes, dtype=np.float32)
        
        # 重塑为原始形状
        share_reshaped = share_array.reshape(-1)
        
        logger.info(f"云端分享数据形状: {share_reshaped.shape}")
        
        # 使用云端模型计算embedding
        text_representation = f"cloud_share_{request.x_coord}"
        
        # 计算embedding
        embedding = embedding_model.compute_embedding(text_representation)
        
        # 将embedding与分享数据结合
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
        
        logger.info(f"云端embedding计算完成: {request.request_id}")
        
        return EmbeddingResponse(
            result=result_b64,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"云端计算embedding时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/merge_embeddings", response_model=MergeResponse)
async def merge_embeddings(request: MergeRequest):
    """
    合并本地和云端的embedding计算结果
    
    Args:
        request: 包含本地和云端计算结果的请求
        
    Returns:
        合并后的embedding和最终推理结果
    """
    try:
        logger.info(f"收到embedding合并请求: {request.request_id}")
        
        # 解码本地结果
        local_bytes = base64.b64decode(request.local_result.encode('utf-8'))
        local_array = np.frombuffer(local_bytes, dtype=np.float32)
        
        # 解码云端结果
        cloud_bytes = base64.b64decode(request.cloud_result.encode('utf-8'))
        cloud_array = np.frombuffer(cloud_bytes, dtype=np.float32)
        
        # 确保形状一致
        if local_array.shape != cloud_array.shape:
            raise ValueError("本地和云端结果形状不一致")
        
        # 使用秘密分享协议重构原始embedding
        shares = [local_array, cloud_array]
        reconstructed_embedding = secret_sharing.reconstruct_embedding(shares, request.x_coords)
        
        logger.info(f"重构的embedding形状: {reconstructed_embedding.shape}")
        
        # 使用推理引擎处理合并后的embedding
        final_result = inference_engine.process_embedding(reconstructed_embedding)
        
        # 编码结果
        merged_bytes = reconstructed_embedding.tobytes()
        merged_b64 = base64.b64encode(merged_bytes).decode('utf-8')
        
        # 存储合并信息
        pending_merges[request.request_id] = {
            "x_coords": request.x_coords,
            "input_shapes": [local_array.shape, cloud_array.shape],
            "output_shape": reconstructed_embedding.shape,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        logger.info(f"embedding合并完成: {request.request_id}")
        
        return MergeResponse(
            merged_embedding=merged_b64,
            final_result=final_result,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"合并embedding时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/inference", response_model=InferenceResponse)
async def perform_inference(request: InferenceRequest):
    """
    执行完整的推理流程
    
    Args:
        request: 包含输入文本的请求
        
    Returns:
        推理结果
    """
    try:
        logger.info(f"收到推理请求: {request.request_id}")
        
        # 1. 将输入文本拆分成秘密分享
        text_shares, x_coords = secret_sharing.split_text(request.text, n_shares=2)
        
        logger.info(f"文本已拆分为 {len(text_shares)} 个分享")
        
        # 2. 发送分享到本地服务器进行计算
        local_url = f"{Config.get_local_url()}/api/v1/compute_embedding"
        local_response = requests.post(
            local_url,
            json={
                "share_data": text_shares[0],
                "x_coord": x_coords[0],
                "request_id": f"{request.request_id}_local"
            },
            timeout=Config.TIMEOUT
        )
        
        if local_response.status_code != 200:
            raise HTTPException(status_code=500, detail="本地计算失败")
        
        local_result = local_response.json()["result"]
        
        # 3. 在云端计算另一个分享
        cloud_share_bytes = base64.b64decode(text_shares[1].encode('utf-8'))
        cloud_share_array = np.frombuffer(cloud_share_bytes, dtype=np.float32)
        
        # 计算云端embedding
        cloud_text = f"cloud_inference_{x_coords[1]}"
        cloud_embedding = embedding_model.compute_embedding(cloud_text)
        cloud_result_embedding = cloud_embedding + cloud_share_array.reshape(cloud_embedding.shape) * 0.1
        
        # 编码云端结果
        cloud_result_bytes = cloud_result_embedding.tobytes()
        cloud_result_b64 = base64.b64encode(cloud_result_bytes).decode('utf-8')
        
        # 4. 合并结果
        merge_response = await merge_embeddings(MergeRequest(
            local_result=local_result,
            cloud_result=cloud_result_b64,
            x_coords=x_coords,
            request_id=f"{request.request_id}_merge"
        ))
        
        logger.info(f"推理完成: {request.request_id}")
        
        return InferenceResponse(
            result=merge_response.final_result,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"执行推理时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def get_status():
    """获取服务器状态"""
    return {
        "server_type": "cloud",
        "active_requests": len(active_requests),
        "pending_merges": len(pending_merges),
        "embedding_model_loaded": embedding_model.model is not None,
        "inference_model_loaded": inference_engine.model is not None,
        "embedding_model_name": embedding_model.model_name,
        "inference_model_name": inference_engine.model_name,
        "device": str(embedding_model.device)
    }

@app.get("/api/v1/requests/{request_id}")
async def get_request_info(request_id: str):
    """获取特定请求的信息"""
    if request_id in active_requests:
        return active_requests[request_id]
    elif request_id in pending_merges:
        return pending_merges[request_id]
    else:
        raise HTTPException(status_code=404, detail="请求未找到")

@app.delete("/api/v1/requests/{request_id}")
async def clear_request(request_id: str):
    """清除特定请求的信息"""
    if request_id in active_requests:
        del active_requests[request_id]
        return {"status": "cleared"}
    elif request_id in pending_merges:
        del pending_merges[request_id]
        return {"status": "cleared"}
    else:
        raise HTTPException(status_code=404, detail="请求未找到")

@app.post("/api/v1/test_inference")
async def test_inference():
    """测试推理功能"""
    try:
        test_text = "这是一个测试推理的文本"
        result = await perform_inference(InferenceRequest(
            text=test_text,
            request_id="test_inference"
        ))
        
        return {
            "status": "success",
            "test_text": test_text,
            "result": result.result
        }
    except Exception as e:
        logger.error(f"测试推理时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("云端计算服务器启动中...")
    Config.create_dirs()
    logger.info(f"服务器将在 http://{Config.CLOUD_HOST}:{Config.CLOUD_PORT} 启动")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("云端计算服务器关闭中...")

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "cloud_server:app",
        host=Config.CLOUD_HOST,
        port=Config.CLOUD_PORT,
        reload=True,
        log_level=Config.LOG_LEVEL.lower()
    ) 