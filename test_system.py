#!/usr/bin/env python3
"""
系统测试脚本
用于测试基于秘密分享协议的分布式大模型推理引擎
"""

import requests
import json
import time
import numpy as np
from typing import Dict, List

from config import Config
from secret_sharing import secret_sharing

def test_secret_sharing():
    """测试秘密分享协议"""
    print("=== 测试秘密分享协议 ===")
    
    # 测试文本拆分和重构
    test_text = "这是一个测试文本，用于验证秘密分享协议的功能。"
    print(f"原始文本: {test_text}")
    
    # 拆分文本
    shares, x_coords = secret_sharing.split_text(test_text, n_shares=2)
    print(f"拆分后的分享数量: {len(shares)}")
    print(f"x坐标: {x_coords}")
    
    # 重构文本
    reconstructed_text = secret_sharing.reconstruct_text(shares, x_coords)
    print(f"重构后的文本: {reconstructed_text}")
    
    # 验证重构是否正确
    if test_text == reconstructed_text:
        print("✓ 文本秘密分享测试通过")
    else:
        print("✗ 文本秘密分享测试失败")
    
    # 测试embedding拆分和重构
    print("\n--- 测试embedding秘密分享 ---")
    test_embedding = np.random.randn(1, 768).astype(np.float32)
    print(f"原始embedding形状: {test_embedding.shape}")
    
    # 拆分embedding
    embedding_shares, embedding_x_coords = secret_sharing.split_embedding(test_embedding, n_shares=2)
    print(f"embedding分享数量: {len(embedding_shares)}")
    
    # 重构embedding
    reconstructed_embedding = secret_sharing.reconstruct_embedding(embedding_shares, embedding_x_coords)
    print(f"重构的embedding形状: {reconstructed_embedding.shape}")
    
    # 验证重构是否正确
    if np.allclose(test_embedding, reconstructed_embedding, atol=1e-6):
        print("✓ embedding秘密分享测试通过")
    else:
        print("✗ embedding秘密分享测试失败")
        print(f"最大差异: {np.max(np.abs(test_embedding - reconstructed_embedding))}")

def test_server_health():
    """测试服务器健康状态"""
    print("\n=== 测试服务器健康状态 ===")
    
    servers = {
        "本地服务器": f"{Config.get_local_url()}",
        "云端服务器": f"{Config.get_cloud_url()}"
    }
    
    for name, url in servers.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {name} 在线")
            else:
                print(f"✗ {name} 响应异常: {response.status_code}")
        except Exception as e:
            print(f"✗ {name} 离线: {e}")

def test_embedding_computation():
    """测试embedding计算功能"""
    print("\n=== 测试embedding计算功能 ===")
    
    # 测试本地embedding计算
    try:
        response = requests.post(f"{Config.get_local_url()}/api/v1/test_embedding")
        if response.status_code == 200:
            result = response.json()
            print("✓ 本地embedding计算测试通过")
            print(f"  测试文本: {result.get('test_text')}")
            print(f"  embedding形状: {result.get('embedding_shape')}")
        else:
            print(f"✗ 本地embedding计算测试失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 本地embedding计算测试失败: {e}")
    
    # 测试云端推理
    try:
        response = requests.post(f"{Config.get_cloud_url()}/api/v1/test_inference")
        if response.status_code == 200:
            result = response.json()
            print("✓ 云端推理测试通过")
            print(f"  测试文本: {result.get('test_text')}")
            print(f"  推理结果: {result.get('result')}")
        else:
            print(f"✗ 云端推理测试失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 云端推理测试失败: {e}")

def test_distributed_inference():
    """测试分布式推理功能"""
    print("\n=== 测试分布式推理功能 ===")
    
    test_texts = [
        "这是一个简单的测试文本。",
        "人工智能技术正在快速发展。",
        "秘密分享协议提供了安全的分布式计算方案。"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}: {text}")
        
        try:
            response = requests.post(
                f"{Config.get_cloud_url()}/api/v1/inference",
                json={
                    "text": text,
                    "request_id": f"test_{i}"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 推理成功")
                print(f"  结果: {result.get('result')}")
            else:
                print(f"✗ 推理失败: {response.status_code}")
                print(f"  错误: {response.text}")
                
        except Exception as e:
            print(f"✗ 推理过程中出错: {e}")

def test_server_status():
    """测试服务器状态查询"""
    print("\n=== 测试服务器状态查询 ===")
    
    servers = {
        "本地服务器": f"{Config.get_local_url()}/api/v1/status",
        "云端服务器": f"{Config.get_cloud_url()}/api/v1/status"
    }
    
    for name, url in servers.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"✓ {name} 状态查询成功")
                print(f"  服务器类型: {status.get('server_type')}")
                print(f"  活跃请求数: {status.get('active_requests', 0)}")
                if 'embedding_model_loaded' in status:
                    print(f"  embedding模型: {'已加载' if status.get('embedding_model_loaded') else '未加载'}")
                if 'inference_model_loaded' in status:
                    print(f"  推理模型: {'已加载' if status.get('inference_model_loaded') else '未加载'}")
            else:
                print(f"✗ {name} 状态查询失败: {response.status_code}")
        except Exception as e:
            print(f"✗ {name} 状态查询失败: {e}")

def run_all_tests():
    """运行所有测试"""
    print("开始系统测试...")
    print("=" * 50)
    
    # 测试秘密分享协议
    test_secret_sharing()
    
    # 测试服务器健康状态
    test_server_health()
    
    # 测试embedding计算
    test_embedding_computation()
    
    # 测试分布式推理
    test_distributed_inference()
    
    # 测试服务器状态
    test_server_status()
    
    print("\n" + "=" * 50)
    print("系统测试完成")

if __name__ == "__main__":
    run_all_tests() 