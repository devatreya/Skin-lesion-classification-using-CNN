#!/usr/bin/env python3
"""
Script to download and organize the HAM10000 dataset for skin lesion classification.
Supports both Kaggle API and manual download methods.
"""

import os
import subprocess
import zipfile
import pandas as pd
from pathlib import Path
import shutil
import json

def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_key = kaggle_dir / "kaggle.json"
    
    if kaggle_key.exists():
        print("Kaggle API credentials found!")
        return True
    else:
        print("Kaggle API credentials not found.")
        print("Please follow these steps to set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/ directory")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_from_kaggle():
    """Download HAM10000 dataset from Kaggle using API."""
    print("Downloading HAM10000 dataset from Kaggle...")
    
    try:
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Download dataset
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "kmader/skin-cancer-mnist-ham10000",
            "-p", str(data_dir),
            "--unzip"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Dataset downloaded successfully from Kaggle!")
            return True
        else:
            print("Error downloading from Kaggle: {}".format(result.stderr))
            return False
            
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it with: pip install kaggle")
        return False
    except Exception as e:
        print("Error downloading from Kaggle: {}".format(e))
        return False

def download_with_requests():
    """Download dataset using direct HTTP requests (fallback method)."""
    print("Direct HTTP download not available for this dataset.")
    print("Please download the dataset manually from Kaggle.")
    return False

def download_ham10000():
    """Download HAM10000 dataset from Kaggle or provide manual instructions."""
    
    # Create data directory structure
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if data already exists
    if (data_dir / "HAM10000_images_part_1").exists() or (data_dir / "ham10000_images").exists():
        print("HAM10000 dataset already exists in data/ folder!")
        return True
    
    # Try Kaggle API first
    if check_kaggle_credentials():
        if download_from_kaggle():
            return True
    
    # If Kaggle download fails, provide manual instructions
    print("\nKaggle download failed. Please download manually:")
    
    instructions = """
    MANUAL DOWNLOAD INSTRUCTIONS:
    
    Method 1 - Kaggle (Recommended):
    1. Go to https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    2. Click 'Download' button (requires Kaggle account)
    3. Extract the downloaded zip file to the data/ folder
    4. The structure should be:
       data/
       ├── HAM10000_images_part_1/
       ├── HAM10000_images_part_2/
       └── HAM10000_metadata.csv
    
    Method 2 - ISIC Archive:
    1. Go to https://challenge.isic-archive.com/
    2. Register for an account
    3. Download the following files:
       - ISIC_2019_Training_Input.zip (images)
       - ISIC_2019_Training_GroundTruth.csv (labels)
    4. Place them in the data/ folder
    5. Run this script again to extract and organize the data
    
    After downloading, run this script again to organize the data.
    """
    
    with open("data/download_instructions.txt", "w") as f:
        f.write(instructions)
    
    print("Download instructions saved to data/download_instructions.txt")
    return False

def organize_data():
    """Organize the downloaded data into proper structure."""
    
    data_dir = Path("data")
    
    # Check if data exists
    if not (data_dir / "HAM10000_images_part_1").exists() and not (data_dir / "ham10000_images").exists():
        print("Data not found. Please download the dataset first.")
        return False
    
    # Create organized directory structure
    organized_dir = data_dir / "organized"
    organized_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different lesion types
    lesion_types = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma', 
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    
    for code, name in lesion_types.items():
        (organized_dir / code).mkdir(exist_ok=True)
    
    print("Data organization structure created.")
    print("Please manually organize images into the appropriate folders based on their labels.")
    
    return True


if __name__ == "__main__":
    print("Setting up data directory structure...")
    
    # Try to download (will likely fail due to authentication)
    download_success = download_ham10000()
    
    if not download_success:
        print("\nPlease download the dataset manually following the instructions above.")
    
    print("\nData setup complete!")
    print("Next steps:")
    print("1. Download the actual dataset following instructions in data/download_instructions.txt")
    print("2. Run the data exploration script to analyze the dataset")
