"""Package setup for Alytes-ReID."""

from setuptools import find_packages, setup

setup(
    name="alytes-reid",
    version="0.1.0",
    description="Individual re-identification of midwife toads (Alytes obstetricans)",
    author="danort92",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.1.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "reid": [
            "pytorch-metric-learning>=2.3.0",
            "timm>=0.9.0",
            "faiss-cpu>=1.7.4",
        ],
        "ui": [
            "gradio>=4.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mlflow>=2.8.0",
        ],
    },
)
