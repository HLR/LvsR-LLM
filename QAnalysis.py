import argparse
import pandas as pd
import numpy as np
import os
from utils import create_file_name, extract_answer
from reader import read_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Dataset Evaluation Test Script for OpenAI Models.")
    parser.add_argument("--datasets", nargs="+", default=["Insurance_Cost", "Used_Car_Prices"],help="List of datasets to evaluate.")
    parser.add_argument("--models", nargs="+",default=["gpt-4-0125-preview", "gpt-3.5-turbo-0125", "meta/meta-llama-3-70b-instruct"],help="List of LLM models to use.")
    parser.add_argument("--in-context-numbers", nargs="+", type=int, default=[0, 10, 30, 100],help="List of in-context example numbers to use.")
    parser.add_argument("--feature-nums", nargs="+", type=int, default=[1, 2, 3],help="List of feature numbers to use.")
    parser.add_argument("--input-folder", type=str, default="LLM_Results",help="The folder's name to read the LLM results.")
    parser.add_argument("--output-folder", type=str, default="./", help="The output folder's name to save the outputs.")
    parser.add_argument("--testing-sampling", type=int, default=0, help="A number assigned to the outputs as sampling.")
    return parser.parse_args()

def get_predictions(dataset, model_name, in_context, feature_num, config, args):
    file_name = create_file_name(
        args.input_folder, dataset, model_name, in_context, feature_num, config,
        testing_sampling=args.testing_sampling
    )
    _, _, _, _, y_test = read_dataset(dataset, config)
    df_test = pd.read_csv(file_name)
    real_responses = []
    predicted_responses = []

    for index, row in df_test.iterrows():
        predicted_response = extract_answer(str(row['processed_response']))
        if predicted_response is not None and predicted_response != -1:
            real_responses.append(y_test[index])
            predicted_responses.append(predicted_response)
    return real_responses, predicted_responses

def evaluate_named_vs_anonymized(dataset, model_name, in_context, feature_num, args):
    ground_truth_named, named_preds = get_predictions(dataset, model_name, in_context, feature_num, "Named_Features", args)
    ground_truth_anon, anon_preds = get_predictions(dataset, model_name, in_context, feature_num, "Anonymized_Features", args)

    min_len = min(len(named_preds), len(anon_preds))
    named_preds = named_preds[:min_len]
    anon_preds = anon_preds[:min_len]
    ground_truth = ground_truth_named[:min_len]

    anon_errors = np.abs(np.array(anon_preds) - np.array(ground_truth))
    named_errors = np.abs(np.array(named_preds) - np.array(ground_truth))

    # ratio = (|anon - true| - |named - true|) / |anon - true|
    ratios = []
    for anon_err, named_err in zip(anon_errors, named_errors):
        if anon_err == 0:
            ratios.append(0 if named_err == 0 else 1)
        else:
            ratio = (anon_err - named_err) / anon_err
            ratios.append(ratio)
    avg_ratio = np.median(ratios) * 100

    shorten_llm_names = {
        "gpt-4-0125-preview": 'GPT-4',
        "gpt-3.5-turbo-0125": 'GPT-3',
        "meta/meta-llama-3-70b-instruct": 'LLaMA 3'
    }.get

    return {
        "Dataset": dataset,
        "Model": shorten_llm_names(model_name),
        "In-Context Examples": in_context,
        "Features": feature_num,
        "Improvement %": f"{avg_ratio:.1f}%"
    }

def main():
    args = parse_arguments()
    results = []

    for dataset in args.datasets:
        for model_name in args.models:
            for in_context in args.in_context_numbers:
                for feature_num in args.feature_nums:
                    # Skip combinations that don't make sense
                    if ((dataset == "Admission_Chance" and in_context > 101) or
                        (feature_num == 4 and dataset != "Used_Car_Prices") or
                        (feature_num == 4 and model_name == "meta/meta-llama-3-70b-instruct") or
                        (in_context == 0)):
                        continue

                    result = evaluate_named_vs_anonymized(
                        dataset, model_name, in_context, feature_num, args
                    )
                    results.append(result)
                    print(f"Processing: {dataset}, {model_name}, {in_context} examples, {feature_num} features")
    results.sort(key=lambda x: float(x["Improvement %"][:-1]))
    df_results = pd.DataFrame(results)
    output_csv = os.path.join(args.output_folder, "QAnalysis_ratios_median.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    main()
