#!/usr/bin/env python3
"""
Training script for skin lesion classification CNN models.
Supports multiple architectures and comprehensive training pipeline.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.cnn_models import SkinLesionCNN, create_ensemble_model
from preprocessing.data_preprocessing import SkinLesionPreprocessor

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.preprocessor = None
        self.model = None
        self.history = None
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        
        for dir_path in [self.models_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def setup_data(self):
        """Setup data preprocessing and generators."""
        print("Setting up data preprocessing...")
        
        self.preprocessor = SkinLesionPreprocessor(
            data_path=self.config['data_path'],
            target_size=tuple(self.config['input_shape'][:2]),
            batch_size=self.config['batch_size']
        )
        
        # Load metadata
        self.preprocessor.load_metadata()
        
        # Create data generators
        self.train_gen, self.val_gen, self.test_gen, self.class_names = \
            self.preprocessor.create_tf_data_generators(
                validation_split=self.config['validation_split'],
                test_split=self.config['test_split'],
                use_augmentation=self.config['use_augmentation'],
                binary_classification=self.config['binary_classification']
            )
        
        print(f"Data setup complete. Classes: {self.class_names}")
        return self.train_gen, self.val_gen, self.test_gen
    
    def build_model(self):
        """Build the specified model architecture."""
        print(f"Building {self.config['architecture']} model...")
        
        cnn = SkinLesionCNN(
            input_shape=tuple(self.config['input_shape']),
            num_classes=len(self.class_names),
            learning_rate=self.config['learning_rate']
        )
        
        # Build model based on architecture
        if self.config['architecture'] == 'simple_cnn':
            self.model = cnn.build_simple_cnn(
                dropout_rate=self.config.get('dropout_rate', 0.5)
            )
        elif self.config['architecture'] == 'resnet50':
            self.model = cnn.build_resnet50(
                pretrained=self.config.get('pretrained', True),
                fine_tune_layers=self.config.get('fine_tune_layers', 50)
            )
        elif self.config['architecture'] == 'vgg16':
            self.model = cnn.build_vgg16(
                pretrained=self.config.get('pretrained', True)
            )
        elif self.config['architecture'] == 'inception_v3':
            self.model = cnn.build_inception_v3(
                pretrained=self.config.get('pretrained', True)
            )
        elif self.config['architecture'] == 'efficientnet':
            self.model = cnn.build_efficientnet(
                pretrained=self.config.get('pretrained', True)
            )
        else:
            raise ValueError(f"Unknown architecture: {self.config['architecture']}")
        
        # Get model info
        info = cnn.get_model_info()
        print(f"Model built successfully!")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        
        return self.model
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        model_path = self.models_dir / f"{self.config['architecture']}_best.h5"
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 15),
            restore_best_weights=True,
            verbose=1
        ))
        
        # Learning rate reduction
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ))
        
        # CSV logger
        csv_path = self.logs_dir / f"{self.config['architecture']}_training.csv"
        callbacks.append(CSVLogger(str(csv_path)))
        
        # TensorBoard
        tb_path = self.logs_dir / f"{self.config['architecture']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        callbacks.append(TensorBoard(
            log_dir=str(tb_path),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ))
        
        return callbacks
    
    def train_model(self):
        """Train the model."""
        print("Starting model training...")
        
        # Calculate steps per epoch
        steps_per_epoch = self.train_gen.samples // self.config['batch_size']
        validation_steps = self.val_gen.samples // self.config['batch_size']
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Get class weights if available
        class_weights = self.preprocessor.class_weights
        
        # Train model
        self.history = self.model.fit(
            self.train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config['epochs'],
            validation_data=self.val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        print("Evaluating model on test data...")
        
        # Calculate test steps
        test_steps = self.test_gen.samples // self.config['batch_size']
        
        # Evaluate model
        test_results = self.model.evaluate(
            self.test_gen,
            steps=test_steps,
            verbose=1
        )
        
        # Get metric names
        metric_names = self.model.metrics_names
        
        # Print results
        print("\nTest Results:")
        for name, value in zip(metric_names, test_results):
            print(f"{name}: {value:.4f}")
        
        # Save results
        results = dict(zip(metric_names, test_results))
        results_path = self.output_dir / f"{self.config['architecture']}_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_training_history(self):
        """Plot and save training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        print("Plotting training history...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{self.config['architecture']}_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history plot saved to {plot_path}")
    
    def save_model(self):
        """Save the trained model."""
        model_path = self.models_dir / f"{self.config['architecture']}_final.h5"
        self.model.save(str(model_path))
        print(f"Model saved to {model_path}")
        
        # Also save model architecture as JSON
        arch_path = self.models_dir / f"{self.config['architecture']}_architecture.json"
        with open(arch_path, 'w') as f:
            json.dump(self.model.to_json(), f, indent=2)
        
        return model_path
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("="*50)
        print("SKIN LESION CLASSIFICATION TRAINING PIPELINE")
        print("="*50)
        
        # Setup data
        self.setup_data()
        
        # Build model
        self.build_model()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        test_results = self.evaluate_model()
        
        # Plot training history
        self.plot_training_history()
        
        # Save model
        model_path = self.save_model()
        
        # Save configuration
        config_path = self.output_dir / f"{self.config['architecture']}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("\n" + "="*50)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model saved to: {model_path}")
        print(f"Results saved to: {self.output_dir}")
        print(f"Test accuracy: {test_results.get('accuracy', 'N/A'):.4f}")
        
        return test_results

def get_default_config():
    """Get default training configuration."""
    return {
        'data_path': 'data',
        'output_dir': 'outputs',
        'architecture': 'resnet50',
        'input_shape': [224, 224, 3],
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'test_split': 0.1,
        'use_augmentation': True,
        'binary_classification': True,
        'pretrained': True,
        'fine_tune_layers': 50,
        'dropout_rate': 0.5,
        'early_stopping_patience': 15
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train skin lesion classification model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--architecture', type=str, 
                       choices=['simple_cnn', 'resnet50', 'vgg16', 'inception_v3', 'efficientnet'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.architecture:
        config['architecture'] = args.architecture
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.pretrained:
        config['pretrained'] = True
    if args.no_augmentation:
        config['use_augmentation'] = False
    
    # Create trainer and run pipeline
    trainer = ModelTrainer(config)
    results = trainer.run_training_pipeline()
    
    return results

if __name__ == "__main__":
    main()
