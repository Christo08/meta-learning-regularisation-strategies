# Understanding Regularisation in Neural Networks through Meta Learning

## Overview
This project explores the impact of regularisation techniques in neural networks using meta learning. It provides tools for dataset creation, model training, optimisation, and statistical analysis, leveraging PyTorch and popular machine learning libraries.

## Features
- Modular codebase for neural networks, decision trees, random forests, and SVMs
- Meta learning trainer and optimisers for various models
- Dataset creation and statistics calculation utilities
- Menu-driven interface for easy experimentation
- Support for CUDA-enabled hardware

## Installation
1. Clone the repository:
   ```powershell
   git clone <repo-url>
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
Run the main program:
```powershell
python main.py
```
Follow the menu prompts to:
- Optimise neural networks
- Create average NN settings
- Create dataset instances
- Get statistics of meta learning datasets
- Optimise and train meta learning models

## Project Structure
- `main.py`: Entry point, menu interface
- `InstanceCreator/`: Dataset creation utilities
- `Models/`: Model definitions and datasets
- `ModelTrainer/`: Trainers for different models
- `Optimisers/`: Model optimisation scripts
- `Utils/`: Utility functions (file handling, stats, menus, etc.)
- `Data/`: Data, checkpoints, and settings

## Dependencies
See `requirements.txt` for a full list. Key packages:
- PyTorch
- scikit-learn
- numpy, pandas
- matplotlib, seaborn
- imbalanced-learn
- networkx