import pandas as pd
import numpy as np

def shuffle_and_split_data(dataset):
    df = pd.read_csv(f"{dataset}/dataset.csv")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if dataset == "Admission_Chance":
        df_incontext = df_shuffled.head(100).reset_index(drop=True)
        df_test = df_shuffled.tail(300).reset_index(drop=True)
    elif dataset == "Used_Car_Prices":
        df_incontext = df_shuffled.head(450).reset_index(drop=True)
        df_test = df_shuffled.tail(302).reset_index(drop=True)
    elif dataset == "Insurance_Cost":
        df_incontext = df_shuffled.head(500).reset_index(drop=True)
        df_test = df_shuffled.tail(300).reset_index(drop=True)
    
    df_shuffled.to_csv(f"{dataset}/dataset_shuffled.csv", index=False)
    df_incontext.to_csv(f"{dataset}/incontext.csv", index=False)
    df_test.to_csv(f"{dataset}/test.csv", index=False)
    
    print(f"Data shuffled and split. Files saved: {dataset}/dataset_shuffled.csv, {dataset}/incontext.csv, {dataset}/test.csv")

def preprocess_usedcars(df):
    df = df[df['franchise_make'].isin(['Toyota', 'Maserati'])]
    df = df[['city_fuel_economy', 'mileage', 'body_type', 'price', 'franchise_make']]
    df = df.dropna()

    df_toyota = df[df['franchise_make'] == 'Toyota'].sample(n=380, random_state=1, replace=True)
    df_maserati = df[df['franchise_make'] == 'Maserati'].sample(n=380, random_state=1, replace=True)
    df = pd.concat([df_toyota, df_maserati])

    passenger_cars = ['Convertible', 'Coupe', 'Hatchback', 'Sedan']
    df['passenger_car'] = df['body_type'].apply(lambda x: 1 if x in passenger_cars else 0)
    df = df[['city_fuel_economy', 'mileage', 'passenger_car', 'price', 'franchise_make']]
    df['franchise_make'] = df['franchise_make'].apply(lambda x: 1 if x == "Toyota" else 0)

    return df