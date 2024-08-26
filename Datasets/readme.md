# Dataset Preprocessing and Machine Learning Model Training

This section provides tools for preprocessing datasets and training machine learning models on them. It includes functionality for shuffling and splitting data, as well as training and evaluating various regression models.

## Datasets

This project uses the following datasets:

1. `Insurance_Cost`[^1]
2. `Admission_Chance`[^2]
3. `Used_Car_Prices`[^3]

---

[^1]: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
[^2]: [Graduate Admissions](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)  
[^3]: [US Used Cars Dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)  
    *The used car price dataset is huge and as a result a small section of it is randomly selected and saved as `dataset.csv`.*

## Files

- `main.py`: The main script that starts data preprocessing and model training processes.
- `data_preprocessing.py`: Contains functions for shuffling and splitting datasets.
- `model_training.py`: Implements the training and evaluation of machine learning models.
- `utils.py`: Utility functions for generating random outputs and mean predictions.

## Usage

The main script (`main.py`) supports two actions: shuffling data and training models.

### Shuffling Data

To shuffle and split a dataset:

```
python main.py --action shuffle --dataset <dataset_name>
```

This will create three files in the dataset's directory:
- `dataset_shuffled.csv`: The entire dataset shuffled.
- `incontext.csv`: A subset of the data for in-context learning.
- `test.csv`: A subset of the data for testing.

### Training Models

To train and evaluate models on a dataset:

```
python main.py --action train --dataset <dataset_name> --output <output_file.csv>
```

This will train several models (Ridge Regression, Decision Tree, ...) on the dataset and evaluate their performance. The results will be saved to the specified output CSV file.

`<dataset_name>` can be one of:

- `Insurance_Cost` https://www.kaggle.com/datasets/mirichoi0218/insurance
- `Admission_Chance` https://www.kaggle.com/datasets/mohansacharya/graduate-admissions
- `Used_Car_Prices` https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset

## Requirements

See the `requirements.txt` file for a list of required Python packages.