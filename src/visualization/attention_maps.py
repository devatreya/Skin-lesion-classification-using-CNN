#!/usr/bin/env python3
"""
Attention maps and model interpretability visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import cv2
from pathlib import Path
import seaborn as sns

class AttentionVisualizer:
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or ['Benign', 'Malignant']
        
    def generate_gradcam(self, img_array, layer_name=None, class_idx=None):
        """Generate Grad-CAM visualization."""
        # Get the last convolutional layer
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Convolutional layer
                    layer_name = layer.name
                    break
        
        # Create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, class_idx]
        
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def generate_saliency_map(self, img_array, class_idx=None):
        """Generate saliency map using gradients."""
        img_tensor = tf.Variable(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_output = predictions[0, class_idx]
        
        # Get gradients
        gradients = tape.gradient(class_output, img_tensor)
        
        # Compute saliency map
        saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
        saliency = saliency[0]  # Remove batch dimension
        
        return saliency.numpy()
    
    def visualize_attention(self, img_array, img_path=None, save_path=None):
        """Visualize different attention mechanisms."""
        # Generate attention maps
        gradcam = self.generate_gradcam(img_array)
        saliency = self.generate_saliency_map(img_array)
        
        # Get prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(img_array[0])
        axes[0, 0].set_title(f'Original Image\nPredicted: {self.class_names[predicted_class]} ({confidence:.3f})')
        axes[0, 0].axis('off')
        
        # Grad-CAM
        axes[0, 1].imshow(img_array[0])
        axes[0, 1].imshow(gradcam, cmap='jet', alpha=0.6)
        axes[0, 1].set_title('Grad-CAM')
        axes[0, 1].axis('off')
        
        # Saliency Map
        axes[0, 2].imshow(saliency, cmap='hot')
        axes[0, 2].set_title('Saliency Map')
        axes[0, 2].axis('off')
        
        # Overlay visualizations
        axes[1, 0].imshow(img_array[0])
        axes[1, 0].imshow(gradcam, cmap='jet', alpha=0.4)
        axes[1, 0].set_title('Grad-CAM Overlay')
        axes[1, 0].axis('off')
        
        # Heatmap only
        axes[1, 1].imshow(gradcam, cmap='jet')
        axes[1, 1].set_title('Grad-CAM Heatmap')
        axes[1, 1].axis('off')
        
        # Saliency overlay
        axes[1, 2].imshow(img_array[0])
        axes[1, 2].imshow(saliency, cmap='hot', alpha=0.4)
        axes[1, 2].set_title('Saliency Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'gradcam': gradcam,
            'saliency': saliency,
            'prediction': predicted_class,
            'confidence': confidence
        }
    
    def compare_attention_methods(self, img_array, save_path=None):
        """Compare different attention visualization methods."""
        # Generate different attention maps
        gradcam = self.generate_gradcam(img_array)
        saliency = self.generate_saliency_map(img_array)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original image
        axes[0, 0].imshow(img_array[0])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grad-CAM
        im1 = axes[0, 1].imshow(gradcam, cmap='jet')
        axes[0, 1].set_title('Grad-CAM')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Saliency Map
        im2 = axes[0, 2].imshow(saliency, cmap='hot')
        axes[0, 2].set_title('Saliency Map')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Combined
        combined = (gradcam + saliency) / 2
        im3 = axes[0, 3].imshow(combined, cmap='viridis')
        axes[0, 3].set_title('Combined')
        axes[0, 3].axis('off')
        plt.colorbar(im3, ax=axes[0, 3])
        
        # Overlay visualizations
        axes[1, 0].imshow(img_array[0])
        axes[1, 0].imshow(gradcam, cmap='jet', alpha=0.5)
        axes[1, 0].set_title('Grad-CAM Overlay')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img_array[0])
        axes[1, 1].imshow(saliency, cmap='hot', alpha=0.5)
        axes[1, 1].set_title('Saliency Overlay')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(img_array[0])
        axes[1, 2].imshow(combined, cmap='viridis', alpha=0.5)
        axes[1, 2].set_title('Combined Overlay')
        axes[1, 2].axis('off')
        
        # Prediction confidence
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        axes[1, 3].bar(self.class_names, predictions[0])
        axes[1, 3].set_title(f'Predictions\nBest: {self.class_names[predicted_class]} ({confidence:.3f})')
        axes[1, 3].set_ylabel('Confidence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def create_attention_visualization_pipeline(model_path, data_generator, output_dir="attention_visualizations"):
    """Create a complete attention visualization pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Create visualizer
    visualizer = AttentionVisualizer(model)
    
    # Process sample images
    for i, (batch_x, batch_y) in enumerate(data_generator):
        if i >= 5:  # Process only first 5 batches
            break
            
        for j in range(min(2, batch_x.shape[0])):  # 2 images per batch
            img = batch_x[j:j+1]
            true_label = np.argmax(batch_y[j])
            
            # Generate visualizations
            save_path = output_dir / f"attention_batch_{i}_img_{j}.png"
            visualizer.visualize_attention(img, save_path=save_path)
            
            print(f"Processed batch {i}, image {j}")

def main():
    """Main function for testing attention visualization."""
    print("Attention visualization module loaded successfully!")
    print("Use AttentionVisualizer class to generate attention maps for trained models.")

if __name__ == "__main__":
    main()
