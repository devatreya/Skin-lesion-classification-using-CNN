#!/usr/bin/env python3
"""
Training script for skin lesion classification CNN models.
Supports multiple architectures and comprehensive training pipeline.
"""

import os
import sys
import argparse
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from datetime import datetime

# Set random seeds for reproducibility ✅
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
print("✅ Random seeds set for reproducibility (seed=42)")

# ✅ Configure GPU memory growth (prevents OOM spikes)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU memory growth enabled for {gpu.name}")
            except RuntimeError as e:
                print(f"⚠️ Could not set memory growth for {gpu.name}: {e}")
except Exception as e:
    print(f"ℹ️ GPU configuration skipped: {e}")

# Enable mixed precision for faster training on Ampere+ GPUs ✅
try:
    from tensorflow.keras import mixed_precision
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        mixed_precision.set_global_policy("mixed_float16")
        print("✅ Mixed precision (float16) enabled for GPU training")
    else:
        print("ℹ️ No GPU detected, using default float32 precision")
except Exception as e:
    print(f"ℹ️ Mixed precision not available: {e}")

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
        """Setup data preprocessing and generators using lesion-level split."""
        print("Setting up data preprocessing...")
        
        self.preprocessor = SkinLesionPreprocessor(
            data_path=self.config['data_path'],
            target_size=tuple(self.config['input_shape'][:2]),
            batch_size=self.config['batch_size']
        )
        
        # Create data generators from split CSV (lesion-level split)
        self.train_gen, self.val_gen, self.test_gen, \
        self.train_steps, self.val_steps, self.test_steps = \
            self.preprocessor.create_generators_from_split(
                split_csv_path=None,  # Will use default: data/labels_binary_split.csv
                use_augmentation=self.config['use_augmentation'],
                architecture=self.config['architecture'],  # Pass architecture for proper preprocessing
                use_sample_weights=self.config.get('use_sample_weights', False)  # ✅ Pass sample_weights flag
            )
        
        # Set class names for binary classification
        self.class_names = ['Benign', 'Malignant']
        
        print(f"Data setup complete. Classes: {self.class_names}")
        print(f"pos_weight for handling imbalance: {self.preprocessor.pos_weight:.3f}")
        
        return self.train_gen, self.val_gen, self.test_gen
    
    def build_model(self):
        """Build the specified model architecture."""
        print(f"Building {self.config['architecture']} model...")
        
        cnn = SkinLesionCNN(
            input_shape=tuple(self.config['input_shape']),
            num_classes=len(self.class_names),
            learning_rate=self.config['learning_rate'],
            pos_weight=self.preprocessor.pos_weight,
            binary_classification=self.config['binary_classification']
        )
        
        # Store CNN wrapper for two-phase training ✅
        self.cnn_wrapper = cnn
        
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
        
        # Model checkpointing - monitor PR-AUC (best for imbalanced data)
        model_path = self.models_dir / f"{self.config['architecture']}_best.h5"
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_pr_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
        
        # Early stopping - monitor PR-AUC
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_pr_auc',
            mode='max',
            patience=self.config.get('early_stopping_patience', 5),
            restore_best_weights=True,
            verbose=1
        ))
        
        # Learning rate reduction - monitor PR-AUC
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_pr_auc',
            mode='max',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
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
        
        # Use steps from preprocessor
        steps_per_epoch = self.train_steps
        validation_steps = self.val_steps
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Setup callbacks (including confusion matrix logging)
        callbacks = self.setup_callbacks()
        
        # Add custom callbacks
        from tensorflow.keras.callbacks import Callback
        import numpy as np
        
        # ✅ Learning Rate Logger (compatible with all TF versions)
        class LrLogger(Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Try learning_rate first (newer TF), fallback to lr (older TF)
                lr_attr = getattr(self.model.optimizer, 'learning_rate', 
                                 getattr(self.model.optimizer, 'lr', None))
                if lr_attr is not None:
                    lr = float(tf.keras.backend.get_value(lr_attr))
                    print(f"\n[Epoch {epoch+1}] Learning Rate: {lr:.6g}")
                else:
                    print(f"\n[Epoch {epoch+1}] Learning Rate: N/A")
        
        class ConfusionMatrixCallback(Callback):
            def __init__(self, val_gen, val_steps, output_dir):
                super().__init__()
                self.val_gen = val_gen
                self.val_steps = val_steps
                self.output_dir = output_dir
            
            def on_epoch_end(self, epoch, logs=None):
                from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
                print(f"\n{'='*60}")
                print(f"VALIDATION METRICS - EPOCH {epoch + 1}")
                print(f"{'='*60}")
                
                # Collect predictions (iterate properly through validation set)
                y_true = []
                y_prob = []
                
                # Create a fresh iterator for the validation generator
                val_iter = iter(self.val_gen)
                for _ in range(self.val_steps):
                    try:
                        x_val, y_val = next(val_iter)
                        preds = self.model.predict(x_val, verbose=0)
                        # Convert TensorFlow tensors to NumPy arrays ✅
                        y_val_np = y_val.numpy() if hasattr(y_val, 'numpy') else y_val
                        preds_np = preds.numpy() if hasattr(preds, 'numpy') else preds
                        y_true.extend(y_val_np.astype(int).flatten())
                        y_prob.extend(preds_np.flatten())
                    except StopIteration:
                        break
                
                y_true = np.array(y_true)
                y_prob = np.array(y_prob)
                y_pred = (y_prob > 0.5).astype(int)
                
                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                print("\nConfusion Matrix:")
                print(f"              Predicted")
                print(f"              Benign  Malignant")
                print(f"Actual Benign    {cm[0][0]:5d}    {cm[0][1]:5d}")
                print(f"       Malignant {cm[1][0]:5d}    {cm[1][1]:5d}")
                
                # Calculate additional metrics
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # ✅ Predicted positive rate - catches constant-class predictions
                pos_rate = (y_pred == 1).mean()
                
                print(f"\nSensitivity (Recall): {sensitivity:.4f}")
                print(f"Specificity: {specificity:.4f}")
                print(f"Predicted Positive Rate: {pos_rate:.4f} (should be ~0.20 for balanced predictions)")
                
                # ROC-AUC
                if len(np.unique(y_true)) > 1:
                    roc_auc = roc_auc_score(y_true, y_prob)
                    print(f"ROC-AUC: {roc_auc:.4f}")
                
                # Print classification report
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred, 
                                           target_names=['Benign', 'Malignant'],
                                           digits=4))
                print(f"{'='*60}\n")
        
        # ✅ Save validation probabilities callback for threshold tuning
        class SaveValProbsCallback(Callback):
            def __init__(self, val_gen, val_steps, output_dir):
                super().__init__()
                self.val_gen = val_gen
                self.val_steps = val_steps
                self.output_dir = output_dir
            
            def on_train_end(self, logs=None):
                """Save validation probabilities after training completes."""
                print(f"\n{'='*60}")
                print("SAVING VALIDATION PROBABILITIES FOR THRESHOLD TUNING")
                print(f"{'='*60}")
                
                y_true = []
                y_prob = []
                
                val_iter = iter(self.val_gen)
                for step in range(self.val_steps):
                    try:
                        x_val, y_val = next(val_iter)
                        preds = self.model.predict(x_val, verbose=0)
                        # Convert TensorFlow tensors to NumPy arrays ✅
                        y_val_np = y_val.numpy() if hasattr(y_val, 'numpy') else y_val
                        preds_np = preds.numpy() if hasattr(preds, 'numpy') else preds
                        y_true.extend(y_val_np.astype(int).flatten())
                        y_prob.extend(preds_np.flatten())
                    except StopIteration:
                        break
                
                # Save to npz file
                prob_path = self.output_dir / 'val_probs.npz'
                np.savez(prob_path,
                        y_true=np.array(y_true),
                        y_prob=np.array(y_prob))
                
                print(f"✅ Saved {len(y_true)} validation probabilities to {prob_path}")
                print(f"   Use: python3 scripts/tune_threshold.py --run_dir {self.output_dir}")
                print(f"{'='*60}\n")
        
        callbacks.append(ConfusionMatrixCallback(self.val_gen, self.val_steps, self.output_dir))
        callbacks.append(SaveValProbsCallback(self.val_gen, self.val_steps, self.output_dir))
        callbacks.append(LrLogger())  # ✅ Add LR logger
        
        # ✅ Compute class weights - MUTUALLY EXCLUSIVE with sample_weights
        w_pos = self.preprocessor.pos_weight
        use_sample_weights = self.config.get('use_sample_weights', False)
        
        if use_sample_weights:
            # Sample weights from dataset (3rd output tensor)
            class_weights = None
            print(f"\n{'='*60}")
            print(f"USING SAMPLE WEIGHTS FROM DATASET")
            print(f"{'='*60}")
            print(f"Benign weight: 1.000")
            print(f"Malignant weight: {w_pos:.3f}")
            print(f"Note: class_weight=None (weights from dataset)")
            print(f"{'='*60}\n")
        else:
            # Use class_weight parameter (standard approach)
            class_weights = {0: 1.0, 1: float(w_pos)}
            print(f"\n{'='*60}")
            print(f"CLASS WEIGHTS FOR TRAINING")
            print(f"{'='*60}")
            print(f"Benign (class 0): {class_weights[0]:.3f}")
            print(f"Malignant (class 1): {class_weights[1]:.3f}")
            print(f"{'='*60}\n")
        
        # Train model
        self.history = self.model.fit(
            self.train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config['epochs'],
            validation_data=self.val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,  # ✅ None if using sample_weights, dict otherwise
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        print("Evaluating model on test data...")
        
        # Use test_steps calculated during setup
        test_steps = self.test_steps
        
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
        
        # Check if history has the required keys
        if not hasattr(self.history, 'history') or not self.history.history:
            print("Training history is empty or incomplete.")
            return
        
        print("Plotting training history...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Check if we have accuracy data
        if 'accuracy' in self.history.history and len(self.history.history['accuracy']) > 0:
            axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.history.history and len(self.history.history['val_accuracy']) > 0:
                axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        else:
            print("No accuracy data available for plotting.")
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Check if we have loss data
        if 'loss' in self.history.history and len(self.history.history['loss']) > 0:
            axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history and len(self.history.history['val_loss']) > 0:
                axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        else:
            print("No loss data available for plotting.")
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        if 'precision' in self.history.history and len(self.history.history['precision']) > 0:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            if 'val_precision' in self.history.history and len(self.history.history['val_precision']) > 0:
                axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        else:
            print("No precision data available for plotting.")
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot recall
        if 'recall' in self.history.history and len(self.history.history['recall']) > 0:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            if 'val_recall' in self.history.history and len(self.history.history['val_recall']) > 0:
                axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        else:
            print("No recall data available for plotting.")
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
    
    def save_run_artifacts(self):
        """Save training configuration and metadata for reproducibility."""
        import json
        from datetime import datetime
        
        # Determine preprocessing type based on architecture
        preprocessing_map = {
            'resnet50': 'ResNet50/ImageNet (caffe-style)',
            'vgg16': 'VGG16/ImageNet',
            'inception_v3': 'InceptionV3 ([-1,1] scaling)',
            'efficientnet': 'EfficientNet/EfficientNetV2',
            'simple_cnn': 'Standard normalization (rescale=1./255)'
        }
        
        artifacts = {
            'timestamp': datetime.now().isoformat(),
            'architecture': self.config['architecture'],
            'preprocessing': preprocessing_map.get(self.config['architecture'], 'unknown'),
            'input_shape': self.config['input_shape'],
            'batch_size': self.config['batch_size'],
            'learning_rate': self.config['learning_rate'],
            'epochs_trained': len(self.history.history.get('loss', [])) if self.history else 0,
            'use_augmentation': self.config['use_augmentation'],
            'pretrained': self.config['pretrained'],
            'two_phase_training': self.config.get('two_phase_training', False),
            'phase1_epochs': self.config.get('phase1_epochs', None),
            'phase2_unfreeze_layers': self.config.get('phase2_unfreeze_layers', None),
            'use_sample_weights': self.config.get('use_sample_weights', False),
            'pos_weight': float(self.preprocessor.pos_weight) if hasattr(self.preprocessor, 'pos_weight') else None,
            'class_weights': {int(k): float(v) for k, v in self.preprocessor.class_weights.items()} if hasattr(self.preprocessor, 'class_weights') else None,
            'random_seed': 42,
            'mixed_precision_enabled': len(tf.config.list_physical_devices('GPU')) > 0,
            'early_stopping_patience': self.config.get('early_stopping_patience', 5)
        }
        
        artifacts_path = self.output_dir / 'training_config.json'
        with open(artifacts_path, 'w') as f:
            json.dump(artifacts, f, indent=2)
        print(f"✅ Training artifacts saved to {artifacts_path}")
        
        # Also save model architecture as JSON
        arch_path = self.models_dir / f"{self.config['architecture']}_architecture.json"
        with open(arch_path, 'w') as f:
            json.dump(self.model.to_json(), f, indent=2)
    
    def run_training_pipeline(self):
        """Run the complete training pipeline with optional two-phase fine-tuning."""
        print("="*50)
        print("SKIN LESION CLASSIFICATION TRAINING PIPELINE")
        print("="*50)
        
        # Setup data
        self.setup_data()
        
        # Build model
        cnn_wrapper = self.build_model()
        
        # Check if two-phase training is enabled ✅
        if self.config.get('two_phase_training', False) and self.config.get('pretrained', False):
            print("\n" + "="*60)
            print("TWO-PHASE FINE-TUNING ENABLED")
            print("="*60)
            print(f"Phase 1: Train head only for {self.config.get('phase1_epochs', 5)} epochs")
            print(f"Phase 2: Unfreeze last {self.config.get('phase2_unfreeze_layers', 30)} layers")
            print("="*60 + "\n")
            
            # PHASE 1: Train head only
            print("\n🔥 PHASE 1: TRAINING HEAD ONLY\n")
            phase1_epochs = self.config.get('phase1_epochs', 5)
            original_epochs = self.config['epochs']
            self.config['epochs'] = phase1_epochs
            
            self.train_model()
            
            # PHASE 2: Unfreeze and fine-tune
            print("\n🔥 PHASE 2: FINE-TUNING WITH UNFROZEN LAYERS\n")
            
            # Get the CNN wrapper from build step
            from models.cnn_models import SkinLesionCNN
            
            # Access the cnn object (need to refactor to store it)
            # For now, we'll unfreeze through the model directly
            if hasattr(self, 'cnn_wrapper'):
                success = self.cnn_wrapper.unfreeze_base_model(
                    num_layers=self.config.get('phase2_unfreeze_layers', 30)
                )
                
                if success:
                    # Train for remaining epochs
                    self.config['epochs'] = original_epochs - phase1_epochs
                    self.train_model()
                else:
                    print("Could not unfreeze base model, continuing with frozen backbone")
            else:
                print("⚠️ Cannot access CNN wrapper for unfreezing. Train with frozen backbone.")
            
            # Restore original epoch count
            self.config['epochs'] = original_epochs
        else:
            # Standard single-phase training
            self.train_model()
        
        # Evaluate model
        test_results = self.evaluate_model()
        
        # Plot training history
        self.plot_training_history()
        
        # Save model
        model_path = self.save_model()
        
        # Save run artifacts (config + preprocessing provenance) ✅
        self.save_run_artifacts()
        
        # Save configuration
        config_path = self.output_dir / f"{self.config['architecture']}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("\n" + "="*50)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model saved to: {model_path}")
        print(f"Results saved to: {self.output_dir}")
        
        # Print test metrics if available
        if test_results and 'accuracy' in test_results:
            print(f"Test accuracy: {test_results['accuracy']:.4f}")
        if test_results and 'loss' in test_results:
            print(f"Test loss: {test_results['loss']:.4f}")
        
        return test_results

def get_default_config():
    """Get default training configuration."""
    return {
        'data_path': 'data',
        'output_dir': 'outputs',
        'architecture': 'resnet50',
        'input_shape': [224, 224, 3],
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.0001,  # Reduced from 0.001 (using AdamW)
        'validation_split': 0.15,  # Used only for old method, ignored with split CSV
        'test_split': 0.15,         # Used only for old method, ignored with split CSV
        'use_augmentation': True,
        'binary_classification': True,
        'pretrained': True,
        'fine_tune_layers': 50,
        'dropout_rate': 0.5,
        'early_stopping_patience': 5,  # Reduced from 15
        'two_phase_training': True,  # ✅ NEW: Enable two-phase fine-tuning
        'phase1_epochs': 5,          # ✅ NEW: Train head only
        'phase2_unfreeze_layers': 30, # ✅ NEW: Unfreeze last 30 layers
        'use_sample_weights': False  # ✅ NEW: If True, use sample_weights; if False, use class_weight dict
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
