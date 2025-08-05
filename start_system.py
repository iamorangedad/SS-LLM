#!/usr/bin/env python3
"""
系统启动脚本
用于启动基于秘密分享协议的分布式大模型推理引擎
"""

import subprocess
import time
import signal
import sys
import os
from typing import List, Optional

def start_server(script_name: str, port: int) -> subprocess.Popen:
    """
    启动服务器
    
    Args:
        script_name: 脚本名称
        port: 端口号
        
    Returns:
        进程对象
    """
    try:
        # 启动服务器进程
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"正在启动 {script_name} (端口: {port})...")
        
        # 等待服务器启动
        time.sleep(3)
        
        if process.poll() is None:
            print(f"✓ {script_name} 启动成功")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"✗ {script_name} 启动失败")
            print(f"错误输出: {stderr}")
            return None
            
    except Exception as e:
        print(f"✗ 启动 {script_name} 时出错: {e}")
        return None

def check_server_health(url: str, name: str) -> bool:
    """
    检查服务器健康状态
    
    Args:
        url: 服务器URL
        name: 服务器名称
        
    Returns:
        是否健康
    """
    try:
        import requests
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ {name} 健康检查通过")
            return True
        else:
            print(f"✗ {name} 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ {name} 健康检查失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 基于秘密分享协议的分布式大模型推理引擎 ===")
    print("正在启动系统...")
    print()
    
    # 创建必要的目录
    os.makedirs("model_cache", exist_ok=True)
    
    # 启动本地服务器
    local_process = start_server("local_server.py", 8001)
    if local_process is None:
        print("本地服务器启动失败，退出")
        return
    
    # 启动云端服务器
    cloud_process = start_server("cloud_server.py", 8002)
    if cloud_process is None:
        print("云端服务器启动失败，退出")
        local_process.terminate()
        return
    
    # 等待服务器完全启动
    print("\n等待服务器完全启动...")
    time.sleep(5)
    
    # 健康检查
    print("\n执行健康检查...")
    local_healthy = check_server_health("http://localhost:8001", "本地服务器")
    cloud_healthy = check_server_health("http://localhost:8002", "云端服务器")
    
    if local_healthy and cloud_healthy:
        print("\n✓ 所有服务器启动成功！")
        print("\n系统信息:")
        print("- 本地服务器: http://localhost:8001")
        print("- 云端服务器: http://localhost:8002")
        print("- API文档: http://localhost:8001/docs 和 http://localhost:8002/docs")
        print("\n可以使用以下命令启动客户端:")
        print("python client.py")
        print("\n按 Ctrl+C 停止所有服务器")
        
        try:
            # 保持进程运行
            while True:
                time.sleep(1)
                # 检查进程是否还在运行
                if local_process.poll() is not None:
                    print("本地服务器已停止")
                    break
                if cloud_process.poll() is not None:
                    print("云端服务器已停止")
                    break
                    
        except KeyboardInterrupt:
            print("\n正在停止服务器...")
            
    else:
        print("\n✗ 部分服务器启动失败")
        if local_process:
            local_process.terminate()
        if cloud_process:
            cloud_process.terminate()
        return
    
    # 清理进程
    if local_process:
        local_process.terminate()
        local_process.wait()
    if cloud_process:
        cloud_process.terminate()
        cloud_process.wait()
    
    print("所有服务器已停止")

if __name__ == "__main__":
    main() 