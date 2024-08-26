import argparse
import pandas as pd
import os
import re
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import create_file_name
from reader import read_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Dataset Evaluation Test Script for OpenAI Models.")
    parser.add_argument("--datasets", nargs="+", default=["Insurance_Cost", "Admission_Chance", "Used_Car_Prices"], help="List of datasets to evaluate.")
    parser.add_argument("--models", nargs="+", default=["gpt-4-0125-preview", "gpt-3.5-turbo-0125"], help="List of LLM models to use.")
    parser.add_argument("--in-context-numbers", nargs="+", type=int, default=[0, 10, 30, 100], help="List of in-context example numbers to use.")
    parser.add_argument("--feature-nums", nargs="+", type=int, default=[1, 2, 3, 4], help="List of feature numbers to use.")
    parser.add_argument("--configs", nargs="+", default=["Named_Features", "Anonymized_Features", "Randomized_Ground_Truth", "Reasoning"], help="List of prompt configurations to use.")
    parser.add_argument("--input-folder", type=str, default="LLM_Results", help="The folder's name to read the LLM results.")
    parser.add_argument("--output-folder", type=str, default="./", help="The output folder's name to save the outputs.")
    parser.add_argument("--testing-sampling", type=int, default=0, help="A number assigned to the outputs as sampling.")
    return parser.parse_args()

def extract_answer(response):
    pattern = r'-?\d+(?:\.\d+)?'
    numbers = re.findall(pattern, response.replace(",", ""))
    try:
        return float(numbers[-1])
    except IndexError:
        return None

def evaluate_responses(dataset, model_name, in_context, feature_num, config, args):
    file_name = create_file_name(args.input_folder, dataset, model_name, in_context, feature_num, config, testing_sampling=args.testing_sampling)
    _, _, _, _, y_test = read_dataset(dataset, config)
    
    df_test = pd.read_csv(file_name)
    real_responses = []
    predicted_responses = []

    for index, row in df_test.iterrows():
        predicted_response = extract_answer(str(row['processed_response']))
        if predicted_response is not None and predicted_response != -1:
            real_responses.append(y_test[index])
            if dataset == "Admission_Chance" and (predicted_response > 1.0 or predicted_response < 0):
                predicted_response = max(0, min(1, predicted_response))
            predicted_responses.append(predicted_response)

    mse = mean_squared_error(real_responses, predicted_responses)
    mae = mean_absolute_error(real_responses, predicted_responses)
    r2 = r2_score(real_responses, predicted_responses)

    return {
        "features": feature_num,
        "dataset": dataset,
        "model": model_name,
        "in_context": in_context,
        "config": config,
        "MSE": float(mse),
        "MAE": float(mae),
        "r2": float(r2)
    }

def main():
    args = parse_arguments()
    results = []

    for dataset in args.datasets:
        for model_name in args.models:
            for in_context in args.in_context_numbers:
                for feature_num in args.feature_nums:
                    for config in args.configs:
                        # Skip certain combinations based on experimental constraints
                        if (in_context == 0 and "Named_Features" not in config) or \
                           (config == "Reasoning" and in_context > 0) or \
                           (dataset == "Admission_Chance" and in_context > 101) or \
                           (feature_num == 4 and dataset != "Used_Car_Prices"):
                            continue
                        
                        result = evaluate_responses(dataset, model_name, in_context, feature_num, config, args)
                        results.append(result)
                        print(f"Processed: {result}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_folder, "evaluation_results.csv"), index=False)

if __name__ == "__main__":
    main()