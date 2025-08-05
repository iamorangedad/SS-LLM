#!/usr/bin/env python3
"""
系统演示脚本
展示基于秘密分享协议的分布式大模型推理引擎的完整工作流程
"""

import requests
import json
import time
from datetime import datetime

from config import Config
from secret_sharing import secret_sharing

def print_step(step_num: int, title: str):
    """打印步骤标题"""
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}: {title}")
    print(f"{'='*60}")

def print_substep(title: str):
    """打印子步骤标题"""
    print(f"\n--- {title} ---")

def demo_secret_sharing():
    """演示秘密分享协议"""
    print_step(1, "秘密分享协议演示")
    
    # 原始输入
    original_text = "人工智能技术正在改变世界，秘密分享协议提供了安全的分布式计算方案。"
    print(f"原始输入: {original_text}")
    
    # 拆分秘密
    print_substep("1.1 拆分秘密")
    shares, x_coords = secret_sharing.split_text(original_text, n_shares=2)
    print(f"生成了 {len(shares)} 个分享")
    print(f"x坐标: {x_coords}")
    print(f"分享1长度: {len(shares[0])} 字符")
    print(f"分享2长度: {len(shares[1])} 字符")
    
    # 重构秘密
    print_substep("1.2 重构秘密")
    reconstructed_text = secret_sharing.reconstruct_text(shares, x_coords)
    print(f"重构结果: {reconstructed_text}")
    
    # 验证结果
    if original_text == reconstructed_text:
        print("✓ 秘密分享协议工作正常")
    else:
        print("✗ 秘密分享协议存在问题")
    
    return shares, x_coords

def demo_distributed_computation():
    """演示分布式计算"""
    print_step(2, "分布式计算演示")
    
    # 检查服务器状态
    print_substep("2.1 检查服务器状态")
    servers = {
        "本地服务器": Config.get_local_url(),
        "云端服务器": Config.get_cloud_url()
    }
    
    for name, url in servers.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {name} 在线")
            else:
                print(f"✗ {name} 响应异常")
        except Exception as e:
            print(f"✗ {name} 离线: {e}")
    
    # 测试embedding计算
    print_substep("2.2 测试embedding计算")
    try:
        response = requests.post(f"{Config.get_local_url()}/api/v1/test_embedding")
        if response.status_code == 200:
            result = response.json()
            print("✓ 本地embedding计算正常")
            print(f"  测试文本: {result.get('test_text')}")
            print(f"  embedding形状: {result.get('embedding_shape')}")
        else:
            print(f"✗ 本地embedding计算失败")
    except Exception as e:
        print(f"✗ 本地embedding计算失败: {e}")

def demo_inference_workflow():
    """演示推理工作流程"""
    print_step(3, "完整推理工作流程演示")
    
    test_texts = [
        "今天天气很好，适合出去散步。",
        "人工智能技术正在快速发展，为各行各业带来变革。",
        "秘密分享协议是一种重要的密码学技术，用于保护数据隐私。"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print_substep(f"3.{i} 推理测试 {i}")
        print(f"输入文本: {text}")
        
        try:
            start_time = time.time()
            
            # 发送推理请求
            response = requests.post(
                f"{Config.get_cloud_url()}/api/v1/inference",
                json={
                    "text": text,
                    "request_id": f"demo_{i}_{int(time.time())}"
                },
                timeout=30
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ 推理成功 (耗时: {end_time - start_time:.2f}秒)")
                print(f"  结果: {result.get('result')}")
            else:
                print(f"✗ 推理失败: {response.status_code}")
                print(f"  错误: {response.text}")
                
        except Exception as e:
            print(f"✗ 推理过程中出错: {e}")

def demo_system_status():
    """演示系统状态监控"""
    print_step(4, "系统状态监控演示")
    
    print_substep("4.1 本地服务器状态")
    try:
        response = requests.get(f"{Config.get_local_url()}/api/v1/status")
        if response.status_code == 200:
            status = response.json()
            print(f"服务器类型: {status.get('server_type')}")
            print(f"活跃请求数: {status.get('active_requests', 0)}")
            print(f"模型已加载: {status.get('model_loaded', False)}")
            print(f"模型名称: {status.get('model_name', 'N/A')}")
            print(f"设备: {status.get('device', 'N/A')}")
        else:
            print("无法获取本地服务器状态")
    except Exception as e:
        print(f"获取本地服务器状态失败: {e}")
    
    print_substep("4.2 云端服务器状态")
    try:
        response = requests.get(f"{Config.get_cloud_url()}/api/v1/status")
        if response.status_code == 200:
            status = response.json()
            print(f"服务器类型: {status.get('server_type')}")
            print(f"活跃请求数: {status.get('active_requests', 0)}")
            print(f"待处理合并数: {status.get('pending_merges', 0)}")
            print(f"embedding模型已加载: {status.get('embedding_model_loaded', False)}")
            print(f"推理模型已加载: {status.get('inference_model_loaded', False)}")
            print(f"embedding模型名称: {status.get('embedding_model_name', 'N/A')}")
            print(f"推理模型名称: {status.get('inference_model_name', 'N/A')}")
        else:
            print("无法获取云端服务器状态")
    except Exception as e:
        print(f"获取云端服务器状态失败: {e}")

def demo_security_features():
    """演示安全特性"""
    print_step(5, "安全特性演示")
    
    print_substep("5.1 数据隐私保护")
    print("在分布式推理过程中:")
    print("- 原始输入被拆分成2个分享")
    print("- 每个分享单独发送到不同的服务器")
    print("- 单个分享无法重构原始数据")
    print("- 只有合并后才能得到完整结果")
    
    print_substep("5.2 通信安全")
    print("- 使用加密通信保护数据传输")
    print("- 支持密钥派生和盐值加密")
    print("- 防止中间人攻击")
    
    print_substep("5.3 计算隔离")
    print("- 本地和云端分别处理部分数据")
    print("- 计算过程相互隔离")
    print("- 减少单点故障风险")

def main():
    """主演示函数"""
    print("基于秘密分享协议的分布式大模型推理引擎")
    print("完整系统演示")
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 演示秘密分享协议
        demo_secret_sharing()
        
        # 演示分布式计算
        demo_distributed_computation()
        
        # 演示推理工作流程
        demo_inference_workflow()
        
        # 演示系统状态监控
        demo_system_status()
        
        # 演示安全特性
        demo_security_features()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出错: {e}")

if __name__ == "__main__":
    main() 