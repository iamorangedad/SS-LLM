# Distributed Large Language Model Inference Engine with Secret Sharing Protocol

A secure distributed inference engine for large language models that implements secret sharing protocols to protect data privacy during computation.

## 🚀 Features

- **Secret Sharing Protocol**: Input data is randomly split into 2 shares using Shamir's secret sharing algorithm
- **Distributed Computation**: Local and cloud servers process different shares of the data
- **Secure Merging**: Results are securely combined in the cloud server
- **Privacy Protection**: Individual shares cannot reconstruct the original data
- **Standard Inference**: Follows standard large language model inference workflows
- **Encrypted Communication**: Secure data transmission with key derivation and salt encryption

## 🏗️ Architecture

```
User Input → Secret Sharing → Local Computation → Cloud Computation → Result Merging → Final Output
```

### Core Components

1. **Secret Sharing Protocol** (`secret_sharing.py`)
   - Implements Shamir's secret sharing algorithm
   - Supports text and embedding splitting/reconstruction
   - Provides secure communication utilities

2. **Local Computation Server** (`local_server.py`)
   - Processes one share of the secret data
   - Runs on local environment
   - Port: 8001

3. **Cloud Computation Server** (`cloud_server.py`)
   - Processes the other share of the secret data
   - Handles result merging and final inference
   - Port: 8002

4. **Client Interface** (`client.py`)
   - User-friendly interactive interface
   - Supports distributed inference requests
   - Monitors server status

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Method 1: Using the Startup Script (Recommended)

```bash
python start_system.py
```

### Method 2: Start Servers Separately

```bash
# Terminal 1: Start local server
python local_server.py

# Terminal 2: Start cloud server
python cloud_server.py
```

### Use the Client

```bash
python client.py
```

### Run Tests

```bash
python test_system.py
```

### View Demo

```bash
python demo.py
```

## 🔧 Configuration

### Environment Variables

- `LOCAL_HOST` - Local server host address
- `LOCAL_PORT` - Local server port
- `CLOUD_HOST` - Cloud server host address
- `CLOUD_PORT` - Cloud server port
- `EMBEDDING_MODEL` - Embedding model name
- `INFERENCE_MODEL` - Inference model name
- `LOG_LEVEL` - Logging level

### Configuration File

Main configuration is in `config.py`:

```python
# Server Configuration
LOCAL_HOST = "localhost"
LOCAL_PORT = 8001
CLOUD_HOST = "localhost"
CLOUD_PORT = 8002

# Model Configuration
EMBEDDING_MODEL = "bert-base-uncased"
INFERENCE_MODEL = "gpt2"

# Secret Sharing Configuration
PRIME = 2**31 - 1
SHARE_COUNT = 2
```

## 🔒 Security Features

### 1. Data Privacy Protection

- Input data is randomly split into 2 shares
- Each share is sent to different servers
- Individual shares cannot reconstruct original data
- Only merging produces complete results

### 2. Secure Communication

- Encrypted communication protects data transmission
- Key derivation and salt encryption
- Protection against man-in-the-middle attacks

### 3. Computational Isolation

- Local and cloud servers process separate data portions
- Isolated computation processes
- Reduced single point of failure risk

## 📡 API Endpoints

### Local Server API

- `GET /` - Health check
- `GET /api/v1/status` - Get server status
- `POST /api/v1/compute_embedding` - Compute embedding share
- `POST /api/v1/test_embedding` - Test embedding computation

### Cloud Server API

- `GET /` - Health check
- `GET /api/v1/status` - Get server status
- `POST /api/v1/compute_embedding` - Compute embedding share
- `POST /api/v1/merge_embeddings` - Merge embedding results
- `POST /api/v1/inference` - Execute complete inference
- `POST /api/v1/test_inference` - Test inference functionality

## 💻 Usage Examples

### Basic Usage

```python
from client import DistributedInferenceClient

# Create client
client = DistributedInferenceClient()

# Check server status
status = client.check_servers()
print(f"Local server: {'Online' if status['local'] else 'Offline'}")
print(f"Cloud server: {'Online' if status['cloud'] else 'Offline'}")

# Perform inference
result = client.perform_inference("This is a test text")
print(f"Inference result: {result['result']}")
```

### Direct API Call

```python
import requests

# Send inference request
response = requests.post(
    "http://localhost:8002/api/v1/inference",
    json={
        "text": "Text to infer",
        "request_id": "unique_request_id"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Inference result: {result['result']}")
else:
    print(f"Inference failed: {response.text}")
```

## 🧪 Testing

### Run All Tests

```bash
python test_system.py
```

### Test Components

- Secret sharing protocol functionality
- Server health status
- Embedding computation
- Distributed inference
- Server status monitoring

## 🔍 Monitoring

### Server Status

Access server status via API endpoints:
- Local server: `http://localhost:8001/api/v1/status`
- Cloud server: `http://localhost:8002/api/v1/status`

### Logs

System provides detailed logging including:
- Server startup status
- Request processing
- Error and exception information
- Performance statistics

## 🛠️ Troubleshooting

### Common Issues

1. **Server Startup Failure**
   - Check if ports are occupied
   - Verify dependencies are correctly installed
   - Review error logs

2. **Model Loading Failure**
   - Check network connection
   - Verify model names are correct
   - System will automatically use simplified models as fallback

3. **Inference Request Timeout**
   - Increase timeout configuration
   - Check server load
   - Ensure stable network connection

### Performance Optimization

1. **Model Caching**
   - Models are automatically cached in `model_cache` directory
   - Faster startup after first load

2. **Concurrent Processing**
   - Servers support concurrent requests
   - Asynchronous processing improves performance

3. **Resource Monitoring**
   - Monitor server status via `/api/v1/status` endpoint
   - Track active requests and system load

## 🔧 Development

### Adding New Models

1. Modify model classes in `model_utils.py`
2. Update model names in configuration
3. Test new model functionality

### Customizing Secret Sharing Protocol

1. Inherit from `SecretSharing` class
2. Implement custom splitting and reconstruction methods
3. Update related server code

### Adding New Computation Nodes

1. Create new server scripts
2. Implement corresponding API endpoints
3. Update client and configuration

## 📁 Project Structure

```
SSLLM/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.py             # System configuration
├── secret_sharing.py     # Secret sharing protocol implementation
├── model_utils.py        # Model utilities and inference engine
├── local_server.py       # Local computation server
├── cloud_server.py       # Cloud computation server
├── client.py             # Client interface
├── start_system.py       # System startup script
├── test_system.py        # System testing script
├── demo.py               # Demonstration script
└── USAGE.md              # Detailed usage documentation
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests to improve this project.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Shamir's Secret Sharing algorithm
- FastAPI for web framework
- Transformers library for model support
- Cryptography library for security features

## 📞 Support

For questions, issues, or contributions, please:

1. Check the documentation in `USAGE.md`
2. Review existing issues
3. Create a new issue with detailed information
4. Contact the maintainers

---

**Note**: This is a research prototype. For production use, additional security audits and optimizations are recommended. 