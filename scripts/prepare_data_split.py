#!/usr/bin/env python3
"""
Prepare lesion-level stratified split for HAM10000 dataset.
This script creates train/val/test splits at the lesion level to prevent data leakage.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.data_preprocessing import SkinLesionPreprocessor

def main():
    print("="*80)
    print("HAM10000 LESION-LEVEL DATA SPLIT PREPARATION")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = SkinLesionPreprocessor(data_path="data", target_size=(224, 224))
    
    # Load metadata
    print("\n[Step 1/2] Loading metadata...")
    preprocessor.load_metadata()
    
    if preprocessor.metadata is None or len(preprocessor.metadata) == 0:
        print("ERROR: No metadata loaded. Please ensure HAM10000_metadata.csv exists in data/")
        return
    
    # Create lesion-level split
    print("\n[Step 2/2] Creating lesion-level stratified split...")
    preprocessor.create_lesion_level_split(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        binary_classification=True,
        random_state=42
    )
    
    print("\n" + "="*80)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the split statistics above")
    print("2. Check the generated file: data/labels_binary_split.csv")
    print("3. Run training with: python3 scripts/train_model.py --architecture resnet50")
    print("="*80)

if __name__ == "__main__":
    main()

