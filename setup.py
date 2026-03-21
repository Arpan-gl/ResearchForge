from setuptools import setup, find_packages

setup(
    name="researchforge",
    version="0.1.0",
    description="Topic to trained model — fully automated ML research pipeline",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="ResearchForge",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.31",
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "nbformat>=5.9",
        "nbconvert>=7.0",
        "jupyter>=1.0",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "imbalanced-learn>=0.11",
        "joblib>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
    ],
    extras_require={
        "gnn": [
            "torch>=2.0",
            "torch_geometric>=2.4",
        ],
        "nlp": [
            "transformers>=4.35",
            "datasets>=2.14",
        ],
        "kaggle": [
            "kaggle>=1.5",
        ],
        "mlflow": [
            "mlflow>=2.8",
        ],
        "export": [
            "weasyprint>=60.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
        ],
        "all": [
            "torch>=2.0",
            "torch_geometric>=2.4",
            "transformers>=4.35",
            "datasets>=2.14",
            "kaggle>=1.5",
            "mlflow>=2.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "researchforge=researchforge.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
    ],
)
