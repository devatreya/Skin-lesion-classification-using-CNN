#!/usr/bin/env python3
"""
Data preprocessing pipeline for skin lesion classification.
Handles data loading, augmentation, normalization, and train/val/test splits.
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SkinLesionPreprocessor:
    def __init__(self, data_path="data", target_size=(224, 224), batch_size=32):
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        self.metadata = None
        
        # Define lesion type mappings
        self.lesion_types = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma', 
            'bkl': 'Benign keratosis-like lesions',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }
        
        # Binary classification mapping (benign vs malignant)
        self.binary_mapping = {
            'nv': 0,  # benign
            'bkl': 0,  # benign
            'akiec': 0,  # benign
            'vasc': 0,  # benign
            'df': 0,  # benign
            'mel': 1,  # malignant
            'bcc': 1   # malignant
        }
    
    def load_metadata(self, metadata_file=None):
        """Load dataset metadata."""
        if metadata_file is None:
            # Look for common metadata files
            possible_files = [
                "HAM10000_metadata.csv",
                "ISIC_2019_Training_GroundTruth.csv", 
                "sample_metadata.csv"
            ]
            
            for file in possible_files:
                file_path = self.data_path / file
                if file_path.exists():
                    metadata_file = file_path
                    break
        
        if metadata_file and Path(metadata_file).exists():
            self.metadata = pd.read_csv(metadata_file)
            print(f"Loaded metadata from {metadata_file}")
        else:
            print("No metadata file found. Creating sample metadata...")
            self._create_sample_metadata()
    
    def _create_sample_metadata(self):
        """Create sample metadata for testing purposes."""
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'image_id': [f'ISIC_{i:07d}' for i in range(n_samples)],
            'dx': np.random.choice(list(self.lesion_types.keys()), n_samples, 
                                 p=[0.67, 0.11, 0.10, 0.05, 0.03, 0.02, 0.02]),
            'dx_type': np.random.choice(['histo', 'follow-up', 'consensus'], n_samples, 
                                      p=[0.8, 0.15, 0.05]),
            'age': np.random.normal(50, 20, n_samples).astype(int),
            'sex': np.random.choice(['male', 'female'], n_samples, p=[0.5, 0.5]),
            'localization': np.random.choice(['back', 'lower extremity', 'torso', 'upper extremity', 
                                            'head/neck', 'palms/soles', 'chest', 'abdomen'], 
                                           n_samples)
        }
        
        self.metadata = pd.DataFrame(sample_data)
        print("Created sample metadata for demonstration")
    
    def preprocess_image(self, image_path, augment=False):
        """Preprocess a single image."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Apply augmentation if requested
            if augment:
                image = self._apply_augmentation(image)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def _apply_augmentation(self, image):
        """Apply data augmentation to an image."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Random rotation (small angles)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            image = np.clip((image - 0.5) * contrast_factor + 0.5, 0, 1)
        
        return image
    
    def create_data_generators(self, validation_split=0.2, test_split=0.1, 
                             use_augmentation=True, binary_classification=True):
        """Create data generators for training, validation, and testing."""
        
        if self.metadata is None:
            self.load_metadata()
        
        # Prepare labels
        if binary_classification:
            labels = self.metadata['dx'].map(self.binary_mapping)
            num_classes = 2
            class_names = ['Benign', 'Malignant']
        else:
            labels = self.metadata['dx']
            num_classes = len(self.lesion_types)
            class_names = list(self.lesion_types.values())
        
        # Encode labels
        if binary_classification:
            y_encoded = labels.values
        else:
            y_encoded = self.label_encoder.fit_transform(labels)
        
        # Convert to categorical
        y_categorical = to_categorical(y_encoded, num_classes)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.metadata.index, y_categorical, 
            test_size=test_split, 
            stratify=y_encoded,
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_split/(1-test_split),
            stratify=y_encoded[X_temp],
            random_state=42
        )
        
        print(f"Data split:")
        print(f"- Training: {len(X_train)} samples")
        print(f"- Validation: {len(X_val)} samples")
        print(f"- Test: {len(X_test)} samples")
        
        # Create data generators
        train_generator = self._create_image_generator(
            X_train, y_train, augment=use_augmentation
        )
        val_generator = self._create_image_generator(
            X_val, y_val, augment=False
        )
        test_generator = self._create_image_generator(
            X_test, y_test, augment=False
        )
        
        # Calculate class weights for imbalanced data
        self._calculate_class_weights(y_encoded)
        
        return train_generator, val_generator, test_generator, class_names
    
    def _create_image_generator(self, indices, labels, augment=False):
        """Create a custom data generator."""
        
        def generator():
            while True:
                batch_indices = np.random.choice(indices, self.batch_size, replace=True)
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    # Get image path (assuming images are in organized folders)
                    image_id = self.metadata.iloc[idx]['image_id']
                    
                    # Look for image in various possible locations
                    image_path = None
                    possible_paths = [
                        self.data_path / "organized" / "images" / f"{image_id}.jpg",
                        self.data_path / "HAM10000_images_part_1" / f"{image_id}.jpg",
                        self.data_path / "HAM10000_images_part_2" / f"{image_id}.jpg",
                        self.data_path / "ham10000_images" / f"{image_id}.jpg",
                        self.data_path / "sample" / f"{image_id}.jpg"
                    ]
                    
                    for path in possible_paths:
                        if path.exists():
                            image_path = path
                            break
                    
                    if image_path is None:
                        # Create a dummy image for demonstration
                        dummy_image = np.random.randint(0, 255, (*self.target_size, 3), dtype=np.uint8)
                        image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
                        image = image.astype(np.float32) / 255.0
                    else:
                        image = self.preprocess_image(image_path, augment=augment)
                        if image is None:
                            continue
                    
                    batch_images.append(image)
                    batch_labels.append(labels[idx])
                
                yield np.array(batch_images), np.array(batch_labels)
        
        return generator()
    
    def _calculate_class_weights(self, y_encoded):
        """Calculate class weights for handling imbalanced data."""
        class_counts = Counter(y_encoded)
        total_samples = len(y_encoded)
        
        self.class_weights = {}
        for class_id, count in class_counts.items():
            self.class_weights[class_id] = total_samples / (len(class_counts) * count)
        
        print(f"Class weights: {self.class_weights}")
    
    def create_tf_data_generators(self, validation_split=0.2, test_split=0.1, 
                                use_augmentation=True, binary_classification=True):
        """Create TensorFlow data generators using ImageDataGenerator."""
        
        if self.metadata is None:
            self.load_metadata()
        
        # Prepare data for ImageDataGenerator
        if binary_classification:
            labels = self.metadata['dx'].map(self.binary_mapping)
            num_classes = 2
            class_names = ['Benign', 'Malignant']
        else:
            labels = self.metadata['dx']
            num_classes = len(self.lesion_types)
            class_names = list(self.lesion_types.values())
        
        # Create temporary directory structure for ImageDataGenerator
        temp_dir = self.data_path / "temp_organized"
        temp_dir.mkdir(exist_ok=True)
        
        for class_name in class_names:
            (temp_dir / class_name).mkdir(exist_ok=True)
        
        # Copy images to organized structure (simplified for demo)
        print("Creating organized directory structure for ImageDataGenerator...")
        
        # Create dummy images for demonstration
        for idx, row in self.metadata.iterrows():
            class_id = labels.iloc[idx]
            if binary_classification:
                class_name = 'Benign' if class_id == 0 else 'Malignant'
            else:
                class_name = self.lesion_types[row['dx']]
            
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (*self.target_size, 3), dtype=np.uint8)
            image_path = temp_dir / class_name / f"{row['image_id']}.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
        
        # Data augmentation parameters
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                zoom_range=0.1,
                validation_split=validation_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            temp_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = train_datagen.flow_from_directory(
            temp_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Calculate class weights
        class_counts = train_generator.classes
        unique_classes = np.unique(class_counts)
        total_samples = len(class_counts)
        
        self.class_weights = {}
        for class_id in unique_classes:
            count = np.sum(class_counts == class_id)
            self.class_weights[class_id] = total_samples / (len(unique_classes) * count)
        
        print(f"Class weights: {self.class_weights}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        
        return train_generator, val_generator, class_names
    
    def visualize_data_samples(self, generator, class_names, num_samples=8):
        """Visualize sample images from the data generator."""
        plt.figure(figsize=(15, 10))
        
        for i in range(num_samples):
            batch_x, batch_y = next(generator)
            
            plt.subplot(2, 4, i + 1)
            plt.imshow(batch_x[0])
            plt.title(f"Class: {class_names[np.argmax(batch_y[0])]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('data/sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_data_summary(self):
        """Get summary statistics of the preprocessed data."""
        if self.metadata is None:
            return "No metadata loaded"
        
        summary = {
            'total_samples': len(self.metadata),
            'target_size': self.target_size,
            'batch_size': self.batch_size,
            'lesion_types': self.lesion_types,
            'binary_mapping': self.binary_mapping
        }
        
        if self.class_weights:
            summary['class_weights'] = self.class_weights
        
        return summary

def main():
    """Main function to test the preprocessing pipeline."""
    preprocessor = SkinLesionPreprocessor()
    
    # Load metadata
    preprocessor.load_metadata()
    
    # Create data generators
    train_gen, val_gen, test_gen, class_names = preprocessor.create_tf_data_generators()
    
    # Visualize samples
    preprocessor.visualize_data_samples(train_gen, class_names)
    
    # Print summary
    print("\nData Preprocessing Summary:")
    summary = preprocessor.get_data_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
