import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import generate_random_output, get_mean_prediction

def train_and_evaluate_models(dataset,output):
    dataset_train = pd.read_csv(f"{dataset}/incontext.csv")
    dataset_test = pd.read_csv(f"{dataset}/test.csv")

    if dataset == "Admission_Chance":
        Y_label = "Chance of Admit"
    elif dataset == "Used_Car_Prices":
        Y_label = "price"
    elif dataset == "Insurance_Cost":
        Y_label = "charges"

    results = []    
    for feature_number in range(1,5):
        for sample_num in [10,30,100]:
            for output_status in ["Real","Random"]:
                if feature_number==4 and not dataset=="Used_Car_Prices": continue
                drop_list = get_drop_list(dataset, feature_number, Y_label)
                
                X_train = pd.get_dummies(dataset_train.drop(drop_list, axis=1))[:sample_num]
                X_test = pd.get_dummies(dataset_test.drop(drop_list, axis=1))[:300]
                
                if output_status=="Random":
                    y_train = [generate_random_output(dataset) for _ in range(sample_num)]
                else:
                    y_train = dataset_train[Y_label][:sample_num]
                
                y_test = dataset_test[Y_label][:300]

                models = [
                    ("Ridge", Ridge()),
                    ("DecisionTree", DecisionTreeRegressor(max_depth=3, random_state=43)),
                    ("RandomForest", RandomForestRegressor(n_estimators=1000, max_depth=2, random_state=42)),
                    ("RandomGaussian", lambda _: [generate_random_output(dataset) for _ in range(300)]),
                    ("Mean Model", lambda _: [get_mean_prediction(dataset) for _ in range(300)]),
                ]
                
                for name, model in models:
                    if name in ["Ridge","DecisionTree","RandomForest"]: 
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    else:
                        y_pred = model(X_test)
                    results.append(evaluate_model(name, y_test, y_pred, dataset, feature_number, sample_num, output_status))

    df_output = pd.DataFrame(results)
    if output is None:
        output=dataset+"_MLResults.csv"
    df_output.to_csv(output, index=False)
    print(f"Results saved to {output}")

def get_drop_list(dataset, feature_number, Y_label):
    if dataset == "Admission_Chance":
        base_drop_list = ['Unnamed: 0', 'Serial No.', 'University Rating', 'SOP', 'LOR ', 'research', Y_label]
        if feature_number == 1:
            return base_drop_list + ['gre_score', 'TOEFL Score']
        elif feature_number == 2:
            return base_drop_list + ['TOEFL Score']
        else:
            return base_drop_list
    elif dataset == "Used_Car_Prices":
        feature_list = ["city_fuel_economy", "mileage", "franchise_make", "passenger_car"]
        return ['Unnamed: 0'] + feature_list[feature_number:] + [Y_label]
    elif dataset == "Insurance_Cost":
        base_drop_list = ['Unnamed: 0', 'sex', 'children', 'region', Y_label]
        if feature_number == 1:
            return base_drop_list + ['bmi', 'age']
        elif feature_number == 2:
            return base_drop_list + ['bmi']
        else:
            return base_drop_list

def evaluate_model(name, y_true, y_pred, dataset, feature_number, sample_num, random_output):
    return {
        'features': feature_number,
        'dataset': dataset,
        'model': name,
        'in_context': sample_num,
        'config': random_output,
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

