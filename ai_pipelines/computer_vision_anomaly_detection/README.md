# Computer Vision Anomaly Detection System

An industrial anomaly detection system using state-of-the-art deep learning models to identify defects and anomalies in manufacturing and quality control applications.

## Project Overview

This system implements advanced anomaly detection techniques for computer vision applications:
- **PatchCore model** for feature extraction and memory bank-based anomaly detection
- **EfficientAD** with teacher-student architecture for efficient anomaly detection
- **Vector database integration** (Qdrant) for similarity search and defect classification
- **Multi-class anomaly detection** handling various defect types simultaneously
- **Industrial-grade performance** for manufacturing quality control

## Key Features

### **PatchCore Anomaly Detection**
- Feature extraction from pre-trained vision models
- Memory bank construction for normal sample representation
- Patch-based anomaly scoring for precise defect localization
- Efficient inference for real-time applications

### **EfficientAD Architecture**
- Teacher-student network for knowledge distillation
- Lightweight anomaly detection with high accuracy
- Fast inference suitable for production environments
- Robust performance across different defect types

### **Vector Database Integration**
- Qdrant vector database for efficient similarity search
- Defect classification and categorization
- Scalable storage for large-scale industrial applications
- Real-time query capabilities

## Technical Implementation

### **Technologies Used**
- **Anomalib**: Industrial anomaly detection framework
- **PyTorch**: Deep learning framework
- **Timm**: Pre-trained vision models
- **Qdrant**: Vector database for similarity search
- **OpenCV**: Image processing operations

### **Pipeline Components**
1. **Data Loading**: MVTec dataset integration for industrial images
2. **Feature Extraction**: Pre-trained model feature extraction
3. **Memory Bank**: Normal sample representation building
4. **Anomaly Detection**: Patch-based and teacher-student methods
5. **Vector Storage**: Qdrant integration for defect classification
6. **Evaluation**: AUROC, precision-recall analysis

## Applications

This system is designed for:
- **Manufacturing quality control** for defect detection
- **Industrial inspection** for product quality assurance
- **Semiconductor manufacturing** for chip defect detection
- **Textile industry** for fabric defect identification
- **Food processing** for contamination detection
- **Medical imaging** for anomaly detection in scans

## Performance Metrics

- **High AUROC scores** for accurate anomaly detection
- **Precise defect localization** with patch-based methods
- **Fast inference times** suitable for production lines
- **Robust performance** across different defect types
- **Scalable architecture** for large-scale deployments

## Research Contributions

- **PDN receptive field analysis** for understanding model behavior
- **Mathematical derivations** of patch-based anomaly detection
- **Performance optimization** for industrial applications
- **Vector database integration** for enhanced defect classification

---

## How to Run

To run the code/notebook open in a Jupypter notebook configured IDE or install the notebook Python 
library using pip and run the command: "jupyter notebook" which launches the Juypter Notebook sever 
locally on your device and opens the Jupyer Notebook interface in your web browser using a loopback IP 
address.


In order for the notebook to run, the images from the Grid, Leather, and Tile categories of the MVTec AD dataset must be stored under 'ai_pipelines/computer_vision_anomaly_detection/datasets' with the default directory structure from https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads.

