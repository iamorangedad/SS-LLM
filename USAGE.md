# 使用说明

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动系统

#### 方法一：使用启动脚本（推荐）
```bash
python start_system.py
```

#### 方法二：分别启动服务器
```bash
# 终端1：启动本地服务器
python local_server.py

# 终端2：启动云端服务器
python cloud_server.py
```

### 3. 使用客户端

```bash
python client.py
```

### 4. 运行测试

```bash
python test_system.py
```

### 5. 查看演示

```bash
python demo.py
```

## 系统架构

```
用户输入 → 秘密分享 → 本地计算 → 云端计算 → 结果合并 → 最终输出
```

### 核心组件

1. **秘密分享协议** (`secret_sharing.py`)
   - 实现Shamir秘密分享算法
   - 支持文本和embedding的拆分与重构
   - 提供安全通信功能

2. **本地计算服务器** (`local_server.py`)
   - 处理秘密分享的一部分计算
   - 运行在本地环境
   - 端口：8001

3. **云端计算服务器** (`cloud_server.py`)
   - 处理秘密分享的另一部分计算
   - 负责结果合并和最终推理
   - 端口：8002

4. **客户端接口** (`client.py`)
   - 提供用户友好的交互界面
   - 支持分布式推理请求
   - 监控服务器状态

## API接口

### 本地服务器 API

- `GET /` - 健康检查
- `GET /api/v1/status` - 获取服务器状态
- `POST /api/v1/compute_embedding` - 计算embedding分享
- `POST /api/v1/test_embedding` - 测试embedding计算

### 云端服务器 API

- `GET /` - 健康检查
- `GET /api/v1/status` - 获取服务器状态
- `POST /api/v1/compute_embedding` - 计算embedding分享
- `POST /api/v1/merge_embeddings` - 合并embedding结果
- `POST /api/v1/inference` - 执行完整推理
- `POST /api/v1/test_inference` - 测试推理功能

## 配置选项

### 环境变量

- `LOCAL_HOST` - 本地服务器主机地址
- `LOCAL_PORT` - 本地服务器端口
- `CLOUD_HOST` - 云端服务器主机地址
- `CLOUD_PORT` - 云端服务器端口
- `EMBEDDING_MODEL` - embedding模型名称
- `INFERENCE_MODEL` - 推理模型名称
- `LOG_LEVEL` - 日志级别

### 配置文件

主要配置在 `config.py` 中：

```python
# 服务器配置
LOCAL_HOST = "localhost"
LOCAL_PORT = 8001
CLOUD_HOST = "localhost"
CLOUD_PORT = 8002

# 模型配置
EMBEDDING_MODEL = "bert-base-uncased"
INFERENCE_MODEL = "gpt2"

# 秘密分享配置
PRIME = 2**31 - 1
SHARE_COUNT = 2
```

## 安全特性

### 1. 数据隐私保护

- 输入数据被随机拆分成2份
- 每份数据单独发送到不同服务器
- 单个分享无法重构原始数据
- 只有合并后才能得到完整结果

### 2. 通信安全

- 使用加密通信保护数据传输
- 支持密钥派生和盐值加密
- 防止中间人攻击

### 3. 计算隔离

- 本地和云端分别处理部分数据
- 计算过程相互隔离
- 减少单点故障风险

## 故障排除

### 常见问题

1. **服务器启动失败**
   - 检查端口是否被占用
   - 确认依赖包已正确安装
   - 查看错误日志

2. **模型加载失败**
   - 检查网络连接
   - 确认模型名称正确
   - 系统会自动使用简化模型作为备选

3. **推理请求超时**
   - 增加超时时间配置
   - 检查服务器负载
   - 确认网络连接稳定

### 日志查看

系统会输出详细的日志信息，包括：
- 服务器启动状态
- 请求处理过程
- 错误和异常信息
- 性能统计

### 性能优化

1. **模型缓存**
   - 模型会自动缓存到 `model_cache` 目录
   - 首次加载后后续启动会更快

2. **并发处理**
   - 服务器支持并发请求
   - 使用异步处理提高性能

3. **资源监控**
   - 通过 `/api/v1/status` 接口监控服务器状态
   - 查看活跃请求数和系统负载

## 扩展开发

### 添加新模型

1. 修改 `model_utils.py` 中的模型类
2. 更新配置文件中的模型名称
3. 测试新模型的功能

### 自定义秘密分享协议

1. 继承 `SecretSharing` 类
2. 实现自定义的拆分和重构方法
3. 更新相关服务器代码

### 添加新的计算节点

1. 创建新的服务器脚本
2. 实现相应的API接口
3. 更新客户端和配置

## 示例代码

### 基本使用

```python
from client import DistributedInferenceClient

# 创建客户端
client = DistributedInferenceClient()

# 检查服务器状态
status = client.check_servers()
print(f"本地服务器: {'在线' if status['local'] else '离线'}")
print(f"云端服务器: {'在线' if status['cloud'] else '离线'}")

# 执行推理
result = client.perform_inference("这是一个测试文本")
print(f"推理结果: {result['result']}")
```

### 直接API调用

```python
import requests

# 发送推理请求
response = requests.post(
    "http://localhost:8002/api/v1/inference",
    json={
        "text": "要推理的文本",
        "request_id": "unique_request_id"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"推理结果: {result['result']}")
else:
    print(f"推理失败: {response.text}")
```

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。 