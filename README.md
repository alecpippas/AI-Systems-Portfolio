# AI Systems Portfolio

A portfolio of several AI pipelines (ETL, training, fine-tuning, etc.) and a text-to-video RAG system prototype. Included are several computer vision pipelines for object detection, object segmentation, and anomaly detection. The RAG system takes textual user prompts and returns relevant video clips from corpus of class lecture recordings.

## 🚀 Projects

### [RAG Video Retrieval System](./project/)
A production-ready text-to-video retrieval system that uses fine-tuned CLIP models and vector databases to enable semantic search through video content. The system processes lecture recordings and allows users to find relevant video segments using natural language queries.

**Key Features:**
- Fine-tuned CLIP model for video-text alignment
- Qdrant vector database for efficient similarity search
- ETL pipeline for video processing and feature extraction
- Gradio web interface for user interaction
- Docker containerization for deployment

**Technologies:** PyTorch, CLIP, Qdrant, MongoDB, Gradio, Docker

### [AI Pipelines](./ai_pipelines/)
A collection of modern AI pipelines demonstrating various computer vision and machine learning techniques:

- **Statistical Learning Foundations**: Optimization pipelines using PyTorch for maximum likelihood estimation and gradient descent
- **Computer Vision Anomaly Detection**: Industrial anomaly detection using PatchCore and EfficientAD models
- **Object Detection & Tracking**: YOLO-based object detection with Kalman filter tracking
- **PDDL LLM Routing Agent**: Planning domain definition language framework for LLM-based routing

## 🛠️ Technologies

- **Deep Learning**: PyTorch, CLIP, YOLO, EfficientAD
- **Computer Vision**: OpenCV, Ultralytics, Anomalib
- **Vector Databases**: Qdrant
- **MLOps**: DVC, MLflow, Docker
- **Web Development**: Gradio, FastAPI
- **Planning & Reasoning**: PDDL

## 📁 Repository Structure

```
├── project/                 # RAG Video Retrieval System
│   ├── pipelines/          # ETL, fine-tuning, and inference pipelines
│   ├── llm_engineering/    # Core system components
│   └── demonstration.ipynb # System demonstration
├── ai_pipelines/           # AI Pipeline Projects
│   ├── statistical_learning_foundations/
│   ├── computer_vision_anomaly_detection/
│   ├── object_detection_tracking/
│   └── pddl_llm_routing_agent/
└── docs/                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)
- CUDA-compatible GPU (recommended for training)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd ai-systems-portfolio

# Install dependencies
pip install -r requirements.lock

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running the RAG System
```bash
cd project
docker-compose up -d  # Start Qdrant and MongoDB
rye run python -m pipelines.gradio_app  # Launch web interface
```

## 📊 CI/CD Pipeline

This repository includes automated testing and quality assurance:
- **Code Quality**: Ruff linting and Black formatting
- **Testing**: pytest with coverage reporting
- **Security**: Automated security scanning
- **Deployment**: Docker containerization

## 🤝 Contributing

This is a portfolio repository showcasing AI systems work. For questions or collaboration opportunities, please reach out via GitHub.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with modern AI engineering practices and a focus on production-ready systems.*
