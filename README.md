# Skin Lesion Classification using CNN

A comprehensive deep learning project for classifying skin lesions as benign or malignant using Convolutional Neural Networks (CNNs). This project includes data preprocessing, multiple CNN architectures, training pipelines, evaluation metrics, and visualization tools.

## 🎯 Project Overview

This project implements a complete pipeline for skin lesion classification using the HAM10000 dataset, which contains 10,000 dermoscopic images of 7 different types of skin lesions. The main goal is to distinguish between benign and malignant lesions using state-of-the-art CNN architectures.

## 📁 Project Structure

```
Skin-lesion-classification-using-CNN/
├── data/                           # Dataset directory
│   ├── HAM10000_images_part_1/     # Image data part 1
│   ├── HAM10000_images_part_2/     # Image data part 2
│   └── HAM10000_metadata.csv       # Dataset metadata
├── src/                            # Source code
│   ├── models/                     # CNN model implementations
│   │   ├── cnn_models.py          # Model architectures
│   │   ├── evaluation.py          # Evaluation metrics
│   │   └── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── preprocessing/              # Data preprocessing
│   │   ├── data_exploration.py    # Data analysis
│   │   └── data_preprocessing.py  # Data pipeline
│   └── visualization/              # Visualization tools
│       └── attention_maps.py      # Attention visualization
├── scripts/                        # Executable scripts
│   ├── download_data.py           # Data download
│   └── train_model.py             # Training script
├── outputs/                        # Training outputs
│   ├── models/                    # Saved models
│   ├── logs/                      # Training logs
│   └── plots/                     # Visualization plots
├── requirements.txt               # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Skin-lesion-classification-using-CNN

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download HAM10000 dataset
python3 scripts/download_data.py
```

**Note**: You'll need to download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place it in the `data/` folder.

### 3. Data Exploration

```bash
# Analyze dataset characteristics
python3 src/preprocessing/data_exploration.py
```

### 4. Train Model

```bash
# Train with default ResNet50 architecture
python3 scripts/train_model.py --architecture resnet50 --epochs 50

# Train with different architectures
python3 scripts/train_model.py --architecture vgg16 --epochs 30
python3 scripts/train_model.py --architecture simple_cnn --epochs 40
```

### 5. Evaluate Model

```bash
# Run comprehensive evaluation
python3 -c "
from src.models.evaluation import ModelEvaluator
evaluator = ModelEvaluator('outputs/models/resnet50_best.h5')
# Add evaluation code here
"
```

## 🏗️ Model Architectures

The project supports multiple CNN architectures:

### 1. **Simple CNN**
- Custom architecture with 4 convolutional blocks
- Batch normalization and dropout for regularization
- Global average pooling and dense layers

### 2. **ResNet50** (Recommended)
- Pre-trained on ImageNet
- Transfer learning with fine-tuning
- Excellent performance for medical imaging

### 3. **VGG16**
- Pre-trained VGG16 architecture
- Good baseline for comparison
- Simpler than ResNet but effective

### 4. **InceptionV3**
- Pre-trained Inception architecture
- Multi-scale feature extraction
- Good for diverse image characteristics

### 5. **EfficientNet**
- State-of-the-art efficiency
- Pre-trained EfficientNetB0
- Best accuracy-to-parameters ratio

## 📊 Features

### Data Preprocessing
- **Image resizing** to 224x224 pixels
- **Data augmentation** (rotation, flipping, brightness, contrast)
- **Normalization** to [0,1] range
- **Train/validation/test split** (70/20/10)
- **Class balancing** with weighted sampling

### Training Pipeline
- **Multiple architectures** support
- **Transfer learning** with pre-trained models
- **Early stopping** and learning rate reduction
- **Model checkpointing** for best weights
- **TensorBoard** logging for monitoring
- **Class weights** for imbalanced data

### Evaluation Metrics
- **Accuracy, Precision, Recall, F1-score**
- **ROC-AUC** and **Precision-Recall curves**
- **Confusion matrix** visualization
- **Classification report** with per-class metrics
- **Prediction confidence** analysis

### Visualization Tools
- **Grad-CAM** attention maps
- **Saliency maps** for model interpretability
- **Training history** plots
- **Class distribution** analysis
- **Misclassification** analysis

## 🔧 Usage Examples

### Basic Training

```python
from scripts.train_model import ModelTrainer

# Default configuration
config = {
    'architecture': 'resnet50',
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001
}

trainer = ModelTrainer(config)
results = trainer.run_training_pipeline()
```

### Hyperparameter Tuning

```python
from src.models.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner()
grid_results = tuner.grid_search(param_grid)
random_results = tuner.random_search(param_distributions, n_iter=20)
bayesian_study = tuner.bayesian_optimization(n_trials=50)
```

### Model Evaluation

```python
from src.models.evaluation import ModelEvaluator

evaluator = ModelEvaluator('path/to/model.h5')
evaluator.predict(test_generator)
metrics = evaluator.calculate_metrics()
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curve()
```

### Attention Visualization

```python
from src.visualization.attention_maps import AttentionVisualizer

visualizer = AttentionVisualizer(model)
attention_results = visualizer.visualize_attention(img_array)
```

## 📈 Expected Results

With the HAM10000 dataset and ResNet50 architecture, you can expect:

- **Training Accuracy**: 85-90%
- **Validation Accuracy**: 80-85%
- **Test Accuracy**: 78-83%
- **ROC-AUC**: 0.85-0.90
- **F1-Score**: 0.80-0.85

## 🛠️ Advanced Usage

### Custom Configuration

```python
# Create custom training configuration
config = {
    'architecture': 'resnet50',
    'input_shape': [224, 224, 3],
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.0001,
    'use_augmentation': True,
    'binary_classification': True,
    'pretrained': True,
    'fine_tune_layers': 30
}
```

### Ensemble Models

```python
from src.models.cnn_models import create_ensemble_model

# Load multiple trained models
models = [model1, model2, model3]
ensemble = create_ensemble_model(models)
```

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.12+
- OpenCV 4.8+
- Pandas, NumPy, Matplotlib
- Scikit-learn
- Kaggle API (for data download)
- Optuna (for hyperparameter tuning)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [ISIC Archive](https://challenge.isic-archive.com/)
- TensorFlow/Keras team
- Medical imaging research community

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

## 🔬 Research Applications

This project can be used for:
- Medical image analysis research
- Computer-aided diagnosis systems
- Educational purposes in deep learning
- Benchmarking CNN architectures
- Transfer learning experiments

---

**Note**: This project is for educational and research purposes. For clinical applications, ensure proper validation and regulatory compliance.
