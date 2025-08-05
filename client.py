import requests
import json
import logging
import uuid
from typing import Dict, List, Optional
from datetime import datetime

from config import Config

# 配置日志
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class DistributedInferenceClient:
    """分布式推理客户端"""
    
    def __init__(self):
        """初始化客户端"""
        self.local_url = Config.get_local_url()
        self.cloud_url = Config.get_cloud_url()
        self.session = requests.Session()
        self.session.timeout = Config.TIMEOUT
    
    def check_servers(self) -> Dict[str, bool]:
        """
        检查服务器状态
        
        Returns:
            服务器状态字典
        """
        status = {}
        
        try:
            # 检查本地服务器
            response = self.session.get(f"{self.local_url}/")
            status["local"] = response.status_code == 200
        except Exception as e:
            logger.error(f"本地服务器检查失败: {e}")
            status["local"] = False
        
        try:
            # 检查云端服务器
            response = self.session.get(f"{self.cloud_url}/")
            status["cloud"] = response.status_code == 200
        except Exception as e:
            logger.error(f"云端服务器检查失败: {e}")
            status["cloud"] = False
        
        return status
    
    def get_server_status(self) -> Dict[str, Dict]:
        """
        获取服务器详细状态
        
        Returns:
            服务器状态信息
        """
        status = {}
        
        try:
            # 获取本地服务器状态
            response = self.session.get(f"{self.local_url}/api/v1/status")
            if response.status_code == 200:
                status["local"] = response.json()
            else:
                status["local"] = {"error": "无法获取本地服务器状态"}
        except Exception as e:
            status["local"] = {"error": str(e)}
        
        try:
            # 获取云端服务器状态
            response = self.session.get(f"{self.cloud_url}/api/v1/status")
            if response.status_code == 200:
                status["cloud"] = response.json()
            else:
                status["cloud"] = {"error": "无法获取云端服务器状态"}
        except Exception as e:
            status["cloud"] = {"error": str(e)}
        
        return status
    
    def perform_inference(self, text: str, request_id: Optional[str] = None) -> Dict:
        """
        执行分布式推理
        
        Args:
            text: 输入文本
            request_id: 请求ID，如果为None则自动生成
            
        Returns:
            推理结果
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        try:
            logger.info(f"开始执行分布式推理: {request_id}")
            
            # 发送推理请求到云端服务器
            response = self.session.post(
                f"{self.cloud_url}/api/v1/inference",
                json={
                    "text": text,
                    "request_id": request_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"推理完成: {request_id}")
                return {
                    "status": "success",
                    "request_id": request_id,
                    "input_text": text,
                    "result": result["result"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_msg = f"推理请求失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "request_id": request_id,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            error_msg = f"推理过程中出错: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "request_id": request_id,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def test_embedding_computation(self) -> Dict:
        """
        测试embedding计算功能
        
        Returns:
            测试结果
        """
        results = {}
        
        try:
            # 测试本地embedding计算
            response = self.session.post(f"{self.local_url}/api/v1/test_embedding")
            if response.status_code == 200:
                results["local"] = response.json()
            else:
                results["local"] = {"error": "本地embedding测试失败"}
        except Exception as e:
            results["local"] = {"error": str(e)}
        
        try:
            # 测试云端推理
            response = self.session.post(f"{self.cloud_url}/api/v1/test_inference")
            if response.status_code == 200:
                results["cloud"] = response.json()
            else:
                results["cloud"] = {"error": "云端推理测试失败"}
        except Exception as e:
            results["cloud"] = {"error": str(e)}
        
        return results


def main():
    """主函数 - 提供交互式界面"""
    client = DistributedInferenceClient()
    
    print("=== 基于秘密分享协议的分布式大模型推理引擎 ===")
    print("客户端已启动")
    print()
    
    while True:
        print("\n请选择操作:")
        print("1. 检查服务器状态")
        print("2. 获取详细状态信息")
        print("3. 执行推理")
        print("4. 测试功能")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            print("\n检查服务器状态...")
            status = client.check_servers()
            print(f"本地服务器: {'在线' if status.get('local') else '离线'}")
            print(f"云端服务器: {'在线' if status.get('cloud') else '离线'}")
        
        elif choice == "2":
            print("\n获取详细状态信息...")
            status = client.get_server_status()
            print("本地服务器状态:")
            print(json.dumps(status.get("local", {}), indent=2, ensure_ascii=False))
            print("\n云端服务器状态:")
            print(json.dumps(status.get("cloud", {}), indent=2, ensure_ascii=False))
        
        elif choice == "3":
            text = input("\n请输入要推理的文本: ").strip()
            if text:
                print("正在执行推理...")
                result = client.perform_inference(text)
                print("推理结果:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("文本不能为空")
        
        elif choice == "4":
            print("\n测试功能...")
            test_results = client.test_embedding_computation()
            print("测试结果:")
            print(json.dumps(test_results, indent=2, ensure_ascii=False))
        
        elif choice == "5":
            print("\n退出客户端")
            break
        
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main() 