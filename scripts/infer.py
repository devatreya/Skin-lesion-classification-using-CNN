#!/usr/bin/env python3
"""
Inference script for skin lesion classification.
Loads trained model, applies preprocessing, uses tuned threshold, and outputs predictions.
"""

import sys
from pathlib import Path
import numpy as np
import json
import argparse
import cv2
import pandas as pd
import tensorflow as tf
from glob import glob

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def load_model_and_config(model_path, config_path):
    """Load trained model and its configuration."""
    model = tf.keras.models.load_model(model_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return model, config


def load_threshold(threshold_path):
    """Load optimal threshold from tuning."""
    if Path(threshold_path).exists():
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
        return threshold_data['threshold']
    else:
        print(f"⚠️ Threshold file not found at {threshold_path}, using default 0.5")
        return 0.5


def get_preprocessing_function(architecture):
    """Get the correct preprocessing function for the architecture."""
    if architecture == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    elif architecture == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    elif architecture == 'inception_v3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input
    elif architecture == 'efficientnet':
        try:
            from tensorflow.keras.applications.efficientnet import preprocess_input
            return preprocess_input
        except (ImportError, AttributeError):
            try:
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
                return preprocess_input
            except:
                return lambda x: x / 255.0
    else:
        # Simple CNN: standard normalization
        return lambda x: x / 255.0


def preprocess_image(image_path, target_size, preprocess_func):
    """Load and preprocess a single image."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Convert to float and apply preprocessing
    img = img.astype(np.float32)
    img = preprocess_func(img)
    
    return img


def predict_single_image(model, image_path, config, preprocess_func, threshold):
    """Predict on a single image."""
    target_size = tuple(config['input_shape'][:2])
    
    # Preprocess
    img = preprocess_image(image_path, target_size, preprocess_func)
    img_batch = np.expand_dims(img, axis=0)
    
    # Predict
    prob = model.predict(img_batch, verbose=0)[0][0]
    pred_label = 1 if prob >= threshold else 0
    pred_class = 'Malignant' if pred_label == 1 else 'Benign'
    
    return {
        'image': str(image_path),
        'probability_malignant': float(prob),
        'predicted_label': int(pred_label),
        'predicted_class': pred_class,
        'threshold': float(threshold)
    }


def predict_batch(model, image_paths, config, preprocess_func, threshold, batch_size=32):
    """Predict on multiple images efficiently."""
    results = []
    target_size = tuple(config['input_shape'][:2])
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        for img_path in batch_paths:
            try:
                img = preprocess_image(img_path, target_size, preprocess_func)
                batch_images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
        
        # Predict on batch
        batch_array = np.array(batch_images)
        probs = model.predict(batch_array, verbose=0).flatten()
        
        # Convert to results
        for path, prob in zip(valid_paths, probs):
            pred_label = 1 if prob >= threshold else 0
            pred_class = 'Malignant' if pred_label == 1 else 'Benign'
            
            results.append({
                'image': str(path),
                'probability_malignant': float(prob),
                'predicted_label': int(pred_label),
                'predicted_class': pred_class
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Inference for skin lesion classification')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config JSON')
    parser.add_argument('--threshold', type=str, default=None,
                       help='Path to threshold.json (if not provided, uses 0.5)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images for batch inference')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file for predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.image is None and args.image_dir is None:
        parser.error("Must provide either --image or --image_dir")
    
    print(f"\n{'='*80}")
    print("SKIN LESION CLASSIFICATION INFERENCE")
    print(f"{'='*80}\n")
    
    # Load model and config
    print(f"Loading model from {args.model}...")
    model, config = load_model_and_config(args.model, args.config)
    print(f"✅ Model loaded: {config['architecture']}")
    print(f"   Input shape: {config['input_shape']}")
    
    # Load threshold
    if args.threshold:
        threshold = load_threshold(args.threshold)
        print(f"✅ Threshold loaded: {threshold:.4f}")
    else:
        threshold = 0.5
        print(f"ℹ️ Using default threshold: {threshold:.4f}")
    
    # Get preprocessing function
    preprocess_func = get_preprocessing_function(config['architecture'])
    print(f"✅ Preprocessing: {config.get('preprocessing', config['architecture'])}\n")
    
    # Run inference
    if args.image:
        # Single image inference
        print(f"Processing single image: {args.image}")
        result = predict_single_image(model, args.image, config, preprocess_func, threshold)
        
        print(f"\n{'='*80}")
        print("PREDICTION RESULT")
        print(f"{'='*80}")
        print(f"Image: {result['image']}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Probability (Malignant): {result['probability_malignant']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print(f"{'='*80}\n")
        
        # Save to CSV
        df = pd.DataFrame([result])
        df.to_csv(args.output, index=False)
        print(f"✅ Prediction saved to {args.output}")
    
    else:
        # Batch inference
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(glob(str(Path(args.image_dir) / ext)))
        
        if len(image_paths) == 0:
            print(f"❌ No images found in {args.image_dir}")
            return
        
        print(f"Found {len(image_paths)} images in {args.image_dir}")
        print(f"Processing with batch size {args.batch_size}...\n")
        
        results = predict_batch(model, image_paths, config, preprocess_func,
                               threshold, args.batch_size)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        
        # Print summary
        n_benign = (df['predicted_label'] == 0).sum()
        n_malignant = (df['predicted_label'] == 1).sum()
        
        print(f"\n{'='*80}")
        print("BATCH INFERENCE COMPLETE")
        print(f"{'='*80}")
        print(f"Total images: {len(results)}")
        print(f"Predicted Benign: {n_benign} ({100*n_benign/len(results):.1f}%)")
        print(f"Predicted Malignant: {n_malignant} ({100*n_malignant/len(results):.1f}%)")
        print(f"Results saved to: {args.output}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

