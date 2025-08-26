# AI Pipelines

A collection of modern AI pipelines demonstrating various computer vision and machine learning techniques. Each pipeline showcases different aspects of AI systems engineering, from statistical learning foundations to advanced computer vision applications.

## 📊 Pipeline Projects

### [Statistical Learning Foundations](./statistical_learning_foundations/)
**Optimization and Maximum Likelihood Estimation Pipelines**

Demonstrates fundamental machine learning concepts using PyTorch:
- **Maximum Likelihood Estimation**: Exponential distribution parameter estimation using gradient descent
- **Linear Regression**: SGD optimization for function approximation
- **Statistical Inference**: Confidence intervals and hypothesis testing

**Key Technologies:** PyTorch, NumPy, Matplotlib, Seaborn

**Learning Outcomes:** Understanding of optimization algorithms, statistical inference, and gradient-based learning.

---

### [Computer Vision Anomaly Detection](./computer_vision_anomaly_detection/)
**Industrial Anomaly Detection Using Modern Deep Learning**

Advanced computer vision pipeline for detecting defects in industrial settings:
- **PatchCore Model**: Feature extraction and memory bank-based anomaly detection
- **EfficientAD**: Teacher-student architecture for efficient anomaly detection
- **Vector Database Integration**: Qdrant-based similarity search for defect classification
- **Multi-class Anomaly Detection**: Handling multiple defect types simultaneously

**Key Technologies:** Anomalib, PyTorch, Timm, Qdrant, OpenCV

**Learning Outcomes:** Deep learning for computer vision, anomaly detection architectures, vector database integration.

---

### [Object Detection & Tracking](./object_detection_tracking/)
**Real-time Object Detection with Kalman Filter Tracking**

End-to-end video analysis pipeline combining detection and tracking:
- **YOLO Object Detection**: Real-time object detection using Ultralytics YOLO
- **Kalman Filter Tracking**: State estimation and prediction for object trajectories
- **Video Processing Pipeline**: Frame extraction, processing, and reconstruction
- **Motion Analysis**: Direction and velocity estimation for tracked objects

**Key Technologies:** Ultralytics YOLO, OpenCV, FilterPy, NumPy

**Learning Outcomes:** Real-time computer vision, state estimation, video processing pipelines.

---

### [PDDL LLM Routing Agent](./pddl_llm_routing_agent/)
**Planning Domain Definition Language for LLM-based Routing**

This project currently provides a set of PDDL (Planning Domain Definition Language) files that define the core planning domains and problems for intelligent routing scenarios. These files serve as the foundation for automated planning and reasoning about routing tasks.

While the present implementation focuses on the PDDL specifications, future work will expand this into a full ReAct-style agent that integrates:
- **LLM Integration**: Using large language models to interpret natural language route requests and translate them into planning problems
- **Automated Planning**: Leveraging PDDL and planning solvers for AI-driven route optimization and decision making
- **Reasoning Systems**: Incorporating logical planning, constraint satisfaction, and dynamic environment handling

**Key Technologies (planned):** PDDL, Python Planning Libraries, LLM APIs

**Learning Outcomes:** Foundations of automated planning, logical reasoning, and the groundwork for advanced AI systems integration. Future iterations will demonstrate end-to-end intelligent routing with LLM-powered agents.

## 🛠️ Common Technologies

All pipelines share these foundational technologies:
- **Python 3.11+**: Core programming language
- **PyTorch**: Deep learning framework
- **NumPy/SciPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive development and documentation

## 📁 Pipeline Structure

Each pipeline follows a consistent structure:
```
pipeline_name/
├── README.md              # Project description and setup
├── *.ipynb               # Main implementation notebooks
├── data/                 # Data files and datasets
├── models/               # Trained models and checkpoints
├── results/              # Output files and visualizations
└── requirements.txt      # Pipeline-specific dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended for deep learning pipelines)
- Sufficient RAM for large datasets

### Installation
```bash
# Install base dependencies
pip install -r requirements.lock

# Navigate to specific pipeline
cd ai_pipelines/pipeline_name

# Run the pipeline
jupyter notebook
```

### Running Individual Pipelines

Each pipeline can be run independently:

1. **Statistical Learning**: Basic ML concepts and optimization
2. **Anomaly Detection**: Advanced computer vision applications
3. **Object Tracking**: Real-time video analysis
4. **PDDL Routing**: Automated planning and reasoning

## 📊 Performance Metrics

Each pipeline includes evaluation metrics and performance analysis:
- **Statistical Learning**: Convergence plots, parameter estimation accuracy
- **Anomaly Detection**: AUROC, precision-recall curves, defect classification accuracy
- **Object Tracking**: Detection accuracy, tracking precision, motion analysis
- **PDDL Routing**: Planning success rate, solution optimality, reasoning efficiency

## 🔬 Research Applications

These pipelines demonstrate practical applications in:
- **Industrial Quality Control**: Anomaly detection for manufacturing
- **Autonomous Systems**: Object tracking for robotics
- **Logistics Optimization**: Intelligent routing and planning
- **Statistical Analysis**: Data-driven decision making

---

*Each pipeline represents a complete AI system, from data processing to model deployment, showcasing modern AI engineering practices.*
