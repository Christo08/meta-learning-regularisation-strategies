# Understanding Regularisation in Neural Networks through Meta Learning
This project explores the impact of regularisation techniques in neural networks using meta learning. It provides tools for dataset creation, model training, optimisation, and statistical analysis.

## Overview

## Features
- Modular neural networks, decision trees, random forests, and SVMs
- Meta learning trainer and optimisers for various models
- Dataset creation and statistics calculation utilities
- Menu-driven interface for easy experimentation
- Support for CUDA-enabled hardware

## Installation
1. Clone the repository:
   ```powershell
   git clone https://github.com/Christo08/Understanding_Regularisation_in_NN_through_Meta_Learning
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
- Optimise NN: Gets the optimised hyperparameters for a neural network who's performs is used as targets in the meta lea dataset.
- Create Subsets and instances: Creates subsets of the datasets and instances of thus subsets for the meta learning. This includes calculating the meta-features for the subsets, training the NN and save the results.
- Recreate Subsets:  Recreates instances for a given seed.
- Recreate instances: Recreates instances for a given seed and subset.
- Get Statistics of Meta Learning Dataset: Gets the statistics of the meta learning dataset, including the distribution of meta-features and target values, creating charts and heatmaps.
- Optimise Meta Learning: Gets the optimised hyperparameters for meta learners.
- Train Meta Learning: Trains the meta learners using the created meta learning dataset and saves the results.
- Get Statistics of Meta Learners results

## Project Structure
`project-name/`
│
├──`Data/`
│   ├──`CheckPoints/`: Checkpoints for optimising hyperparameters.
│   ├──`Datasets/`: Created meta learning datasets.
│   │   ├──`Input/`: Folders containing the original datasets, the created subsets and a json file that contains details about the datasets.  
│   │   ├──`Output/`
│   │   │  ├──`Graphs/`: Graphs of the meta learning dataset.
│   │   │  ├──`Modules/`: The meta learning modules.
│   │   │  ├──`Processed/`: Cleaned meta learning dataset.
│   │   │  └──`Raw/`: The raw meta learning dataset.
│   │   └──`Results/`: Results from meta learning experiments.
│   │   │  └──`Modules/`: The meta learning modules.
│   └──`Settings/`: Results from meta learning experiments.
│   │   ├──`BasicNN/`: The meta learner datasets' NN hyperparameters.
│   │   └──`MetaLearners/`: The meta learning modules' hyperparameters.
├──`main.py`: Entry point, menu interface.
├──`requirements.txt`
└──`README.md`

## Adding New Datasets
1. Place the dataset in `Data/Datasets/Input/` with a unique name. Skip this step if the dataset is import from other libraries such as pmlb or others.
2. Update the all_datasets.json file with the following details:
   - `name`: Unique name of the dataset.
   - `type`: Source of the dataset (e.g., "pmlb", "csv"). csv is for download datasets and pmlb is for datasets from the pmlb library.
   - `category_columns`: List of categorical columns in the dataset.
   - `target_column`: The name of the target column in the dataset.
   - `drop_columns`: List of columns to drop from the dataset.
   - `file_path`: The path to the dataset file if the dataset is a csv file. Skip this step if the dataset is from a library.
   - `pmlb_name`: The name of the dataset in the pmlb library only needed if the dataset is from the pmlb library.
## Dependencies
See `requirements.txt` for a full list. Key packages:
- PyTorch
- scikit-learn
- numpy,
- pandas
- matplotlib, 
- seaborn
- imbalanced-learn
- networkx

## Authors
- Christiaan P. Opperman