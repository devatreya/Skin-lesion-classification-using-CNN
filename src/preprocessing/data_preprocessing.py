#!/usr/bin/env python3
"""
Data preprocessing pipeline for skin lesion classification.
Handles data loading, augmentation, normalization, and train/val/test splits.
"""

import os
import json
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
        self.pos_weight = None
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
        # CORRECTED: akiec is pre-cancerous/malignant, not benign
        self.binary_mapping = {
            # Benign (0)
            'nv': 0,    # Melanocytic nevi - benign
            'bkl': 0,   # Benign keratosis - benign
            'vasc': 0,  # Vascular lesions - benign
            'df': 0,    # Dermatofibroma - benign
            # Malignant (1)
            'akiec': 1, # Actinic keratoses (pre-cancerous) - MALIGNANT
            'bcc': 1,   # Basal cell carcinoma - malignant
            'mel': 1    # Melanoma - malignant
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
    
    def create_lesion_level_split(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, 
                                   binary_classification=True, random_state=42):
        """
        Create train/val/test split at LESION level to prevent data leakage.
        Multiple images of the same lesion must stay in the same split.
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        # Apply binary mapping
        if binary_classification:
            self.metadata['label'] = self.metadata['dx'].map(self.binary_mapping)
            
            # ✅ Guard on unmapped dx values - fail fast
            if self.metadata['label'].isna().any():
                missing = self.metadata[self.metadata['label'].isna()]['dx'].value_counts()
                raise ValueError(
                    f"❌ ERROR: Unmapped dx classes found in metadata!\n"
                    f"The following dx values are not in binary_mapping:\n{missing}\n"
                    f"Please update binary_mapping in __init__ to include these classes."
                )
        else:
            self.label_encoder.fit(self.metadata['dx'])
            self.metadata['label'] = self.label_encoder.transform(self.metadata['dx'])
        
        # Check if lesion_id exists
        if 'lesion_id' not in self.metadata.columns:
            print("WARNING: lesion_id not found. Using image-level split (potential leakage).")
            self.metadata['lesion_id'] = self.metadata.index
        
        # Get unique lesions with their labels
        lesion_labels = self.metadata.groupby('lesion_id')['label'].first()
        unique_lesions = lesion_labels.index.values
        lesion_label_values = lesion_labels.values
        
        print(f"\n{'='*60}")
        print("LESION-LEVEL STRATIFIED SPLIT")
        print(f"{'='*60}")
        print(f"Total unique lesions: {len(unique_lesions)}")
        print(f"Total images: {len(self.metadata)}")
        
        # Count class distribution
        from collections import Counter
        class_counts = Counter(lesion_label_values)
        print(f"\nClass distribution (lesion-level):")
        for class_id, count in class_counts.items():
            class_name = "Malignant" if class_id == 1 else "Benign"
            pct = 100 * count / len(lesion_label_values)
            print(f"  {class_name} (class {class_id}): {count} lesions ({pct:.1f}%)")
        
        # First split: train vs (val + test)
        from sklearn.model_selection import train_test_split
        train_lesions, temp_lesions = train_test_split(
            unique_lesions,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=lesion_label_values
        )
        
        # Second split: val vs test
        temp_labels = lesion_labels[temp_lesions].values
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state,
            stratify=temp_labels
        )
        
        # Assign split to each image based on lesion_id
        def assign_split(lesion_id):
            if lesion_id in train_lesions:
                return 'train'
            elif lesion_id in val_lesions:
                return 'val'
            else:
                return 'test'
        
        self.metadata['split'] = self.metadata['lesion_id'].apply(assign_split)
        
        # Print split statistics
        print(f"\n{'='*60}")
        print("SPLIT STATISTICS")
        print(f"{'='*60}")
        for split_name in ['train', 'val', 'test']:
            split_df = self.metadata[self.metadata['split'] == split_name]
            n_lesions = split_df['lesion_id'].nunique()
            n_images = len(split_df)
            n_malignant = (split_df['label'] == 1).sum()
            n_benign = (split_df['label'] == 0).sum()
            pct_malignant = 100 * n_malignant / n_images
            
            print(f"\n{split_name.upper()}:")
            print(f"  Lesions: {n_lesions}")
            print(f"  Images: {n_images}")
            print(f"  Benign: {n_benign} ({100-pct_malignant:.1f}%)")
            print(f"  Malignant: {n_malignant} ({pct_malignant:.1f}%)")
        
        # Calculate class weights based on TRAINING set only
        train_labels = self.metadata[self.metadata['split'] == 'train']['label'].values
        self._calculate_class_weights(train_labels)
        
        # Calculate pos_weight for BCEWithLogitsLoss
        n_neg = (train_labels == 0).sum()
        n_pos = (train_labels == 1).sum()
        self.pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"\n{'='*60}")
        print(f"CLASS IMBALANCE HANDLING")
        print(f"{'='*60}")
        print(f"pos_weight for BCEWithLogitsLoss: {self.pos_weight:.3f}")
        print(f"Class weights: {self.class_weights}")
        print(f"{'='*60}\n")
        
        # Save split CSV for reproducibility
        split_csv_path = self.data_path / "labels_binary_split.csv"
        self.metadata.to_csv(split_csv_path, index=False)
        print(f"✅ Split saved to: {split_csv_path}")
        
        # Save split metadata to JSON for reproducibility ✅
        split_metadata = {
            'random_state': random_state,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'train_lesion_ids': list(train_lesions),
            'val_lesion_ids': list(val_lesions),
            'test_lesion_ids': list(test_lesions),
            'pos_weight': float(self.pos_weight),
            'class_weights': {int(k): float(v) for k, v in self.class_weights.items()},
            'split_counts': {
                'train': {'images': len(self.metadata[self.metadata['split'] == 'train']),
                         'benign': int((self.metadata[self.metadata['split'] == 'train']['label'] == 0).sum()),
                         'malignant': int((self.metadata[self.metadata['split'] == 'train']['label'] == 1).sum())},
                'val': {'images': len(self.metadata[self.metadata['split'] == 'val']),
                       'benign': int((self.metadata[self.metadata['split'] == 'val']['label'] == 0).sum()),
                       'malignant': int((self.metadata[self.metadata['split'] == 'val']['label'] == 1).sum())},
                'test': {'images': len(self.metadata[self.metadata['split'] == 'test']),
                        'benign': int((self.metadata[self.metadata['split'] == 'test']['label'] == 0).sum()),
                        'malignant': int((self.metadata[self.metadata['split'] == 'test']['label'] == 1).sum())}
            }
        }
        
        split_metadata_path = self.data_path / "split_metadata.json"
        with open(split_metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        print(f"✅ Split metadata saved to: {split_metadata_path}")
        
        return self.metadata
    
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
    
    def create_generators_from_split(self, split_csv_path=None, use_augmentation=True, 
                                     architecture='resnet50', use_sample_weights=False):
        """
        Create data generators from pre-split CSV file (lesion-level split).
        This ensures no data leakage between train/val/test sets.
        """
        if split_csv_path is None:
            split_csv_path = self.data_path / "labels_binary_split.csv"
        
        split_csv_path = Path(split_csv_path)
        if not split_csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found at {split_csv_path}. "
                "Run scripts/prepare_data_split.py first!"
            )
        
        # Load the split CSV
        print(f"\n{'='*60}")
        print("LOADING DATA FROM SPLIT CSV")
        print(f"{'='*60}")
        self.metadata = pd.read_csv(split_csv_path)
        print(f"Loaded {len(self.metadata)} images from {split_csv_path}")
        
        # Get pos_weight from training data
        train_labels = self.metadata[self.metadata['split'] == 'train']['label'].values
        n_neg = (train_labels == 0).sum()
        n_pos = (train_labels == 1).sum()
        self.pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        # Calculate class weights
        self._calculate_class_weights(train_labels)
        
        print(f"pos_weight: {self.pos_weight:.3f}")
        
        # Find image directories
        image_dirs = [
            self.data_path / "HAM10000_images_part_1",
            self.data_path / "HAM10000_images_part_2",
            self.data_path / "ham10000_images"
        ]
        
        # Get proper preprocessing function for pretrained models
        preprocess_func = None
        if architecture == 'resnet50':
            from tensorflow.keras.applications.resnet50 import preprocess_input
            preprocess_func = preprocess_input
            print("Using ResNet50 preprocessing (ImageNet mean/std)")
        elif architecture == 'vgg16':
            from tensorflow.keras.applications.vgg16 import preprocess_input
            preprocess_func = preprocess_input
            print("Using VGG16 preprocessing (ImageNet mean/std)")
        elif architecture == 'inception_v3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input
            preprocess_func = preprocess_input
            print("Using InceptionV3 preprocessing")
        elif architecture == 'efficientnet':
            # ✅ Handle both TF versions (efficientnet vs efficientnet_v2)
            try:
                from tensorflow.keras.applications.efficientnet import preprocess_input
                preprocess_func = preprocess_input
                print("Using EfficientNet preprocessing")
            except (ImportError, AttributeError):
                try:
                    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
                    preprocess_func = preprocess_input
                    print("Using EfficientNetV2 preprocessing")
                except Exception:
                    preprocess_func = None
                    print("⚠️ EfficientNet preprocessing not available, using standard normalization")
        else:
            # Simple CNN or unknown: use standard normalization
            preprocess_func = None
            print("Using standard normalization (rescale=1./255)")
        
        # Setup augmentation for training
        if use_augmentation and preprocess_func is None:
            # Standard augmentation with rescaling
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                shear_range=0.1,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        elif use_augmentation and preprocess_func is not None:
            # Pretrained model: use preprocessing_function instead of rescale
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_func,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                shear_range=0.1,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        elif preprocess_func is not None:
            # Pretrained, no augmentation
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
        else:
            # Simple CNN, no augmentation
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # No augmentation for val/test
        if preprocess_func is not None:
            val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
        else:
            val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create custom generator function
        def create_generator(split_name, datagen, shuffle=True):
            split_df = self.metadata[self.metadata['split'] == split_name].copy()
            split_df['image_path'] = None
            
            # Find image paths
            for idx, row in split_df.iterrows():
                image_id = row['image_id']
                for img_dir in image_dirs:
                    img_path = img_dir / f"{image_id}.jpg"
                    if img_path.exists():
                        split_df.at[idx, 'image_path'] = str(img_path)
                        break
            
            # Remove rows without images
            split_df = split_df.dropna(subset=['image_path'])
            
            print(f"\n{split_name.upper()} set:")
            print(f"  Images found: {len(split_df)}")
            print(f"  Benign: {(split_df['label'] == 0).sum()}")
            print(f"  Malignant: {(split_df['label'] == 1).sum()}")
            
            # Create generator
            def generator():
                indices = np.arange(len(split_df))
                while True:
                    if shuffle:
                        np.random.shuffle(indices)
                    
                    for start_idx in range(0, len(split_df), self.batch_size):
                        batch_indices = indices[start_idx:start_idx + self.batch_size]
                        batch_images = []
                        batch_labels = []
                        batch_weights = []  # ✅ For sample_weight fallback
                        missing_images = []  # ✅ Track missing images
                        
                        for idx in batch_indices:
                            row = split_df.iloc[idx]
                            img_path = row['image_path']
                            label = row['label']
                            
                            # Load and preprocess image
                            try:
                                img = cv2.imread(img_path)
                                if img is None:
                                    missing_images.append(row['image_id'])  # ✅ Track missing
                                    continue
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, self.target_size)
                                
                                # Apply augmentation/preprocessing
                                if preprocess_func is not None:
                                    # Pretrained model: use its preprocessing (don't manually normalize)
                                    img = img.astype(np.float32)
                                    if split_name == 'train' and use_augmentation:
                                        img = datagen.random_transform(img)
                                    img = preprocess_func(img)  # Apply model-specific preprocessing
                                else:
                                    # Simple CNN: manual normalization
                                    img = img.astype(np.float32) / 255.0
                                    if split_name == 'train' and use_augmentation:
                                        img = datagen.random_transform(img)
                                
                                batch_images.append(img)
                                batch_labels.append(label)
                                
                                # Compute sample weight ✅
                                if use_sample_weights:
                                    # w = 1.0 for benign, w_pos for malignant
                                    w = 1.0 if label == 0 else self.pos_weight
                                    batch_weights.append(w)
                                
                            except Exception as e:
                                print(f"❌ Error loading {img_path}: {e}")
                                missing_images.append(row['image_id'])
                                continue
                        
                        # Warn about missing images ✅
                        if missing_images:
                            print(f"⚠️ Missing {len(missing_images)} images in batch: {missing_images[:5]}")
                        
                        if len(batch_images) > 0:
                            if use_sample_weights:
                                # Yield with sample weights (3-tuple) ✅
                                yield (
                                    np.array(batch_images), 
                                    np.array(batch_labels, dtype=np.float32).reshape(-1, 1),
                                    np.array(batch_weights, dtype=np.float32)
                                )
                            else:
                                # Standard yield (2-tuple)
                                yield (
                                    np.array(batch_images), 
                                    np.array(batch_labels, dtype=np.float32).reshape(-1, 1)
                                )
            
            # Calculate steps per epoch - use ceil to not drop tail batches ✅
            import math
            steps = math.ceil(len(split_df) / self.batch_size)
            
            # Create tf.data.Dataset
            if use_sample_weights:
                # 3-tuple: images, labels, sample_weights ✅
                output_signature = (
                    tf.TensorSpec(shape=(None, *self.target_size, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.float32)  # sample_weights
                )
            else:
                # 2-tuple: images, labels
                output_signature = (
                    tf.TensorSpec(shape=(None, *self.target_size, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                )
            
            dataset = tf.data.Dataset.from_generator(
                generator,
                output_signature=output_signature
            )
            
            # Add prefetch for performance - hide I/O latency ✅
            AUTOTUNE = tf.data.AUTOTUNE
            dataset = dataset.prefetch(AUTOTUNE)
            
            return dataset, steps, len(split_df)
        
        # Create generators for each split
        train_gen, train_steps, train_samples = create_generator('train', train_datagen, shuffle=True)
        val_gen, val_steps, val_samples = create_generator('val', val_test_datagen, shuffle=False)
        test_gen, test_steps, test_samples = create_generator('test', val_test_datagen, shuffle=False)
        
        # ✅ SPLIT INTEGRITY CHECKS - Fail fast if images are missing
        print(f"\n{'='*60}")
        print("SPLIT INTEGRITY CHECKS")
        print(f"{'='*60}")
        
        expected_train = len(self.metadata[self.metadata['split'] == 'train'])
        expected_val = len(self.metadata[self.metadata['split'] == 'val'])
        expected_test = len(self.metadata[self.metadata['split'] == 'test'])
        
        train_loss = expected_train - train_samples
        val_loss = expected_val - val_samples
        test_loss = expected_test - test_samples
        
        print(f"Train: Expected {expected_train}, Found {train_samples}, Lost {train_loss}")
        print(f"Val:   Expected {expected_val}, Found {val_samples}, Lost {val_loss}")
        print(f"Test:  Expected {expected_test}, Found {test_samples}, Lost {test_loss}")
        
        if train_loss > 0 or val_loss > 0 or test_loss > 0:
            total_loss = train_loss + val_loss + test_loss
            loss_pct = 100 * total_loss / (expected_train + expected_val + expected_test)
            print(f"\n⚠️ WARNING: Lost {total_loss} images total ({loss_pct:.2f}%)")
            
            if loss_pct > 5.0:
                print(f"❌ ERROR: More than 5% of images missing! This may skew class balance.")
                print(f"   Check that image directories are correctly mounted:")
                for img_dir in image_dirs:
                    print(f"   - {img_dir} exists: {img_dir.exists()}")
                raise ValueError(f"Too many missing images ({loss_pct:.1f}%). Aborting.")
        else:
            print("✅ All images found successfully!")
        print(f"{'='*60}\n")
        
        print(f"\n{'='*60}")
        print(f"GENERATORS CREATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Train steps/epoch: {train_steps}")
        print(f"Val steps/epoch: {val_steps}")
        print(f"Test steps/epoch: {test_steps}")
        print(f"{'='*60}\n")
        
        # Store for easy access
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.test_steps = test_steps
        
        return train_gen, val_gen, test_gen, train_steps, val_steps, test_steps
    
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
        
        # For now, use validation generator as test generator
        # In a real scenario, you would have a separate test set
        test_generator = val_generator
        
        print(f"Class weights: {self.class_weights}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Test samples: {test_generator.samples}")
        
        return train_generator, val_generator, test_generator, class_names
    
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
