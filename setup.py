from setuptools import setup, find_packages

setup(
    name="skin-lesion-classification",
    version="0.1.0",
    description="CNN for classifying skin lesions as benign or malignant",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.12.0",
        "keras>=2.12.0",
        "opencv-python>=4.8.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "Pillow>=9.5.0",
        "tqdm>=4.65.0",
        "jupyter>=1.0.0",
    ],
)
