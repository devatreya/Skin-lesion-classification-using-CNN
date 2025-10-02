#!/usr/bin/env python3
"""
Data exploration script for skin lesion classification dataset.
Analyzes dataset distribution, image characteristics, and class balance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataExplorer:
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.metadata = None
        self.image_stats = {}
        
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
            print(f"Metadata shape: {self.metadata.shape}")
        else:
            print("No metadata file found. Creating sample metadata...")
            self._create_sample_metadata()
    
    def _create_sample_metadata(self):
        """Create sample metadata for testing purposes."""
        print("No metadata file found. Please download the HAM10000 dataset first.")
        print("Instructions:")
        print("1. Go to https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("2. Download the dataset")
        print("3. Extract to data/ folder")
        print("4. Run this script again")
        
        # Create empty metadata to prevent errors
        self.metadata = pd.DataFrame(columns=['image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'])
        return
    
    def analyze_class_distribution(self):
        """Analyze the distribution of lesion classes."""
        if self.metadata is None:
            print("No metadata loaded. Please load metadata first.")
            return
        
        print("\n=== CLASS DISTRIBUTION ANALYSIS ===")
        
        # Count classes
        class_counts = self.metadata['dx'].value_counts()
        print("\nClass distribution:")
        print(class_counts)
        
        # Calculate percentages
        class_percentages = (class_counts / len(self.metadata) * 100).round(2)
        print("\nClass percentages:")
        print(class_percentages)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Bar plot
        plt.subplot(2, 2, 1)
        class_counts.plot(kind='bar', color='skyblue')
        plt.title('Class Distribution (Count)')
        plt.xlabel('Lesion Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Pie chart
        plt.subplot(2, 2, 2)
        class_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution (Percentage)')
        plt.ylabel('')
        
        # Benign vs Malignant
        plt.subplot(2, 2, 3)
        benign_classes = ['nv', 'bkl', 'akiec', 'vasc', 'df']
        malignant_classes = ['mel', 'bcc']
        
        benign_count = self.metadata[self.metadata['dx'].isin(benign_classes)].shape[0]
        malignant_count = self.metadata[self.metadata['dx'].isin(malignant_classes)].shape[0]
        
        plt.bar(['Benign', 'Malignant'], [benign_count, malignant_count], 
                color=['lightgreen', 'lightcoral'])
        plt.title('Benign vs Malignant Distribution')
        plt.ylabel('Count')
        
        # Class imbalance ratio
        plt.subplot(2, 2, 4)
        imbalance_ratio = class_counts.max() / class_counts.min()
        plt.text(0.5, 0.5, f'Class Imbalance Ratio:\n{imbalance_ratio:.2f}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.title('Class Imbalance Analysis')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('data/class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_counts
    
    def analyze_demographics(self):
        """Analyze demographic distribution."""
        if self.metadata is None:
            print("No metadata loaded. Please load metadata first.")
            return
        
        print("\n=== DEMOGRAPHIC ANALYSIS ===")
        
        # Age distribution
        print(f"\nAge statistics:")
        print(f"Mean age: {self.metadata['age'].mean():.1f}")
        print(f"Median age: {self.metadata['age'].median():.1f}")
        print(f"Age range: {self.metadata['age'].min()} - {self.metadata['age'].max()}")
        
        # Sex distribution
        sex_counts = self.metadata['sex'].value_counts()
        print(f"\nSex distribution:")
        print(sex_counts)
        
        # Localization distribution
        loc_counts = self.metadata['localization'].value_counts()
        print(f"\nTop 5 lesion localizations:")
        print(loc_counts.head())
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Age distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.metadata['age'], bins=30, color='skyblue', alpha=0.7)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        # Sex distribution
        plt.subplot(2, 3, 2)
        sex_counts.plot(kind='bar', color=['lightblue', 'lightpink'])
        plt.title('Sex Distribution')
        plt.xlabel('Sex')
        plt.ylabel('Count')
        
        # Localization distribution
        plt.subplot(2, 3, 3)
        loc_counts.head(8).plot(kind='bar', color='lightgreen')
        plt.title('Lesion Localization')
        plt.xlabel('Body Part')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Age by class
        plt.subplot(2, 3, 4)
        for class_name in self.metadata['dx'].unique():
            class_data = self.metadata[self.metadata['dx'] == class_name]['age']
            plt.hist(class_data, alpha=0.6, label=class_name, bins=20)
        plt.title('Age Distribution by Class')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Sex by class
        plt.subplot(2, 3, 5)
        sex_class_cross = pd.crosstab(self.metadata['dx'], self.metadata['sex'])
        sex_class_cross.plot(kind='bar', stacked=True)
        plt.title('Sex Distribution by Class')
        plt.xlabel('Lesion Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Sex')
        
        # Class distribution by age groups
        plt.subplot(2, 3, 6)
        self.metadata['age_group'] = pd.cut(self.metadata['age'], 
                                          bins=[0, 30, 50, 70, 100], 
                                          labels=['0-30', '31-50', '51-70', '70+'])
        age_class_cross = pd.crosstab(self.metadata['age_group'], self.metadata['dx'])
        age_class_cross.plot(kind='bar', stacked=True)
        plt.title('Class Distribution by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('data/demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_image_characteristics(self, image_dir=None, sample_size=100):
        """Analyze image characteristics (size, color distribution, etc.)."""
        print("\n=== IMAGE CHARACTERISTICS ANALYSIS ===")
        
        if image_dir is None:
            # Look for common image directories
            possible_dirs = [
                "HAM10000_images_part_1",
                "HAM10000_images_part_2", 
                "ham10000_images",
                "sample"
            ]
            
            for dir_name in possible_dirs:
                dir_path = self.data_path / dir_name
                if dir_path.exists():
                    image_dir = dir_path
                    break
        
        if image_dir is None or not Path(image_dir).exists():
            print("No image directory found. Creating sample analysis...")
            self._create_sample_image_analysis()
            return
        
        # Get sample of images
        image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
        if len(image_files) == 0:
            print("No image files found in directory.")
            return
        
        # Sample images for analysis
        sample_files = np.random.choice(image_files, min(sample_size, len(image_files)), replace=False)
        
        print(f"Analyzing {len(sample_files)} sample images...")
        
        # Analyze image characteristics
        widths, heights, channels = [], [], []
        mean_colors = []
        
        for img_file in sample_files:
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w, c = img.shape
                    heights.append(h)
                    widths.append(w)
                    channels.append(c)
                    
                    # Calculate mean color values
                    mean_color = np.mean(img, axis=(0, 1))
                    mean_colors.append(mean_color)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        # Store statistics
        self.image_stats = {
            'widths': np.array(widths),
            'heights': np.array(heights),
            'channels': np.array(channels),
            'mean_colors': np.array(mean_colors)
        }
        
        # Print statistics
        print(f"\nImage size statistics:")
        print(f"Width - Mean: {np.mean(widths):.1f}, Std: {np.std(widths):.1f}, Range: {np.min(widths)}-{np.max(widths)}")
        print(f"Height - Mean: {np.mean(heights):.1f}, Std: {np.std(heights):.1f}, Range: {np.min(heights)}-{np.max(heights)}")
        print(f"Channels: {np.unique(channels)}")
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Image size distribution
        plt.subplot(2, 3, 1)
        plt.scatter(widths, heights, alpha=0.6)
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Image Size Distribution')
        
        # Width distribution
        plt.subplot(2, 3, 2)
        plt.hist(widths, bins=30, alpha=0.7, color='skyblue')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Frequency')
        plt.title('Width Distribution')
        
        # Height distribution
        plt.subplot(2, 3, 3)
        plt.hist(heights, bins=30, alpha=0.7, color='lightcoral')
        plt.xlabel('Height (pixels)')
        plt.ylabel('Frequency')
        plt.title('Height Distribution')
        
        # Aspect ratio
        plt.subplot(2, 3, 4)
        aspect_ratios = np.array(widths) / np.array(heights)
        plt.hist(aspect_ratios, bins=30, alpha=0.7, color='lightgreen')
        plt.xlabel('Aspect Ratio (W/H)')
        plt.ylabel('Frequency')
        plt.title('Aspect Ratio Distribution')
        
        # Color distribution
        plt.subplot(2, 3, 5)
        mean_colors = np.array(mean_colors)
        plt.scatter(mean_colors[:, 0], mean_colors[:, 1], alpha=0.6, label='B vs G')
        plt.xlabel('Mean Blue Value')
        plt.ylabel('Mean Green Value')
        plt.title('Color Distribution (B vs G)')
        
        # Color distribution 3D
        plt.subplot(2, 3, 6)
        ax = plt.gca()
        ax.scatter(mean_colors[:, 0], mean_colors[:, 1], c=mean_colors[:, 2], 
                  alpha=0.6, cmap='viridis')
        plt.xlabel('Mean Blue Value')
        plt.ylabel('Mean Green Value')
        plt.title('Color Distribution (colored by Red)')
        
        plt.tight_layout()
        plt.savefig('data/image_characteristics_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_sample_image_analysis(self):
        """Create sample image analysis for demonstration."""
        print("Creating sample image analysis...")
        
        # Simulate image statistics
        np.random.seed(42)
        n_images = 100
        
        # Simulate realistic image sizes (dermoscopic images are typically square)
        base_size = 600
        sizes = np.random.normal(base_size, 50, n_images).astype(int)
        sizes = np.clip(sizes, 400, 800)  # Reasonable range
        
        self.image_stats = {
            'widths': sizes,
            'heights': sizes,  # Square images
            'channels': np.full(n_images, 3),  # RGB
            'mean_colors': np.random.uniform(50, 200, (n_images, 3))  # Random color means
        }
        
        print("Sample image analysis created for demonstration.")
    
    def generate_report(self):
        """Generate comprehensive data exploration report."""
        print("\n" + "="*50)
        print("COMPREHENSIVE DATA EXPLORATION REPORT")
        print("="*50)
        
        if self.metadata is not None:
            print(f"\nDataset Overview:")
            print(f"- Total samples: {len(self.metadata)}")
            print(f"- Features: {list(self.metadata.columns)}")
            print(f"- Missing values: {self.metadata.isnull().sum().sum()}")
            
            # Class distribution summary
            class_counts = self.metadata['dx'].value_counts()
            print(f"\nClass Distribution Summary:")
            for class_name, count in class_counts.items():
                percentage = (count / len(self.metadata)) * 100
                print(f"- {class_name}: {count} ({percentage:.1f}%)")
            
            # Benign vs Malignant
            benign_classes = ['nv', 'bkl', 'akiec', 'vasc', 'df']
            malignant_classes = ['mel', 'bcc']
            
            benign_count = self.metadata[self.metadata['dx'].isin(benign_classes)].shape[0]
            malignant_count = self.metadata[self.metadata['dx'].isin(malignant_classes)].shape[0]
            
            print(f"\nBinary Classification Summary:")
            print(f"- Benign: {benign_count} ({(benign_count/len(self.metadata)*100):.1f}%)")
            print(f"- Malignant: {malignant_count} ({(malignant_count/len(self.metadata)*100):.1f}%)")
        
        if self.image_stats:
            print(f"\nImage Characteristics Summary:")
            print(f"- Average size: {np.mean(self.image_stats['widths']):.0f}x{np.mean(self.image_stats['heights']):.0f}")
            print(f"- Size range: {np.min(self.image_stats['widths'])}-{np.max(self.image_stats['widths'])} pixels")
            print(f"- Channels: {np.unique(self.image_stats['channels'])}")
        
        print(f"\nRecommendations:")
        print(f"- Consider data augmentation to handle class imbalance")
        print(f"- Resize images to consistent dimensions for model training")
        print(f"- Use stratified sampling for train/validation/test splits")
        print(f"- Consider transfer learning for better performance")
        
        print("\n" + "="*50)

def main():
    """Main function to run data exploration."""
    explorer = DataExplorer()
    
    # Load metadata
    explorer.load_metadata()
    
    # Run analyses
    explorer.analyze_class_distribution()
    explorer.analyze_demographics()
    explorer.analyze_image_characteristics()
    
    # Generate report
    explorer.generate_report()

if __name__ == "__main__":
    main()
