import pandas as pd
from typing import List, Tuple

def read_dataset(dataset: str, config: str) -> Tuple[List[str], List[List[float]], List[List[float]], List[float], List[float]]:
    """
    Read and process dataset files based on the specified dataset and configuration.

    Args:
        dataset (str): The name of the dataset to read ('Insurance_Cost', 'Admission_Chance', or 'Used_Car_Prices').
        config (str): Configuration string, used to determine if feature names should be anonymized.

    Returns:
        Tuple[List[str], List[List[float]], List[List[float]], List[float], List[float]]: 
            - Names: List of feature names and target variable name
            - X_incontext: List of feature vectors for in-context examples
            - X_test: List of feature vectors for test examples
            - Y_incontext: List of target values for in-context examples
            - Y_test: List of target values for test examples

    Raises:
        ValueError: If an unknown dataset name is provided.
    """
    df_incontext = pd.read_csv(f"Datasets/{dataset}/incontext.csv")
    df_test = pd.read_csv(f"Datasets/{dataset}/test.csv")
    
    dataset_processors = {
        "Insurance_Cost": process_insurance_cost,
        "Admission_Chance": process_admission_chance,
        "Used_Car_Prices": process_used_car_prices
    }
    
    processor = dataset_processors.get(dataset)
    if not processor:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    Names, X_incontext, X_test, Y_incontext, Y_test = processor(df_incontext, df_test)
    
    if "Anonymized_Features" in config:
        Names = [f"Feature{i}" for i in range(len(Names) - 1)] + ["Output"]
    
    return Names, X_incontext, X_test, Y_incontext, Y_test

def process_insurance_cost(df_incontext: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[List[str], List[List[float]], List[List[float]], List[float], List[float]]:
    """Process the Insurance Cost dataset."""
    Names = ['smoker', 'bmi', 'age', "annual individual medical costs billed by health insurance in the USA"]
    
    X_incontext = [[float(row['smoker'] == "yes"), float(row['bmi']), float(row['age'])] for _, row in df_incontext.iterrows()]
    Y_incontext = [float(row['charges']) for _, row in df_incontext.iterrows()]
    
    X_test = [[float(row['smoker'] == "yes"), float(row['bmi']), float(row['age'])] for _, row in df_test.iterrows()]
    Y_test = [float(row['charges']) for _, row in df_test.iterrows()]
    
    return Names, X_incontext, X_test, Y_incontext, Y_test

def process_admission_chance(df_incontext: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[List[str], List[List[float]], List[List[float]], List[float], List[float]]:
    """Process the Admission Chance dataset."""
    Names = ['Cumulative GPA', 'GRE Score', 'TOEFL Score', "Chance of Admition"]
    
    X_incontext = [[float(row['cgpa']), float(row['gre_score']), float(row['TOEFL Score'])] for _, row in df_incontext.iterrows()]
    Y_incontext = [float(row['Chance of Admit']) for _, row in df_incontext.iterrows()]
    
    X_test = [[float(row['cgpa']), float(row['gre_score']), float(row['TOEFL Score'])] for _, row in df_test.iterrows()]
    Y_test = [round(float(row['Chance of Admit']), 2) for _, row in df_test.iterrows()]
    
    return Names, X_incontext, X_test, Y_incontext, Y_test

def process_used_car_prices(df_incontext: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[List[str], List[List[float]], List[List[float]], List[float], List[float]]:
    """Process the Used Car Prices dataset."""
    Names = ['City Fuel Economy', 'Mileage', 'Is Toyota', "Is Small (ex. Sedan is small and SUV is large)", 
             "the price of a used car that can be either a Toyota or Maserati in 2019"]
    
    X_incontext = [[float(row['city_fuel_economy']), float(row['mileage']), float(row['franchise_make']), float(row['passenger_car'])] 
                   for _, row in df_incontext.iterrows()]
    Y_incontext = [float(row['price']) for _, row in df_incontext.iterrows()]
    
    X_test = [[float(row['city_fuel_economy']), float(row['mileage']), float(row['franchise_make']), float(row['passenger_car'])] 
              for _, row in df_test.iterrows()]
    Y_test = [float(row['price']) for _, row in df_test.iterrows()]
    
    return Names, X_incontext, X_test, Y_incontext, Y_test