import argparse
import os
import time
import pandas as pd
from openai import OpenAI

from utils import create_file_name, create_IO_example, get_additional_instruction, create_explanation, process_response
from reader import read_dataset

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Dataset Evaluation Script for OpenAI Models.")
    parser.add_argument("--datasets", nargs="+", default=["Insurance_Cost", "Admission_Chance", "Used_Car_Prices"], help="List of datasets to evaluate.")
    parser.add_argument("--models", nargs="+", default=["gpt-4-0125-preview", "gpt-3.5-turbo-0125"], help="List of LLM models to use.")
    parser.add_argument("--in-context-numbers", nargs="+", type=int, default=[0, 10, 30, 100], help="List of in-context example numbers to use.")
    parser.add_argument("--feature-nums", nargs="+", type=int, default=[1, 2, 3, 4], help="List of feature numbers to use.")
    parser.add_argument("--configs", nargs="+", default=["Named_Features", "Anonymized_Features", "Randomized_Ground_Truth", "Reasoning"], choices=["Named_Features", "Anonymized_Features", "Randomized_Ground_Truth", "Reasoning", "Missing_Inputs", "Missing_Inputs_and_Anonymized_Features"], help="List of prompt configurations to use.")    
    parser.add_argument("--api-key-token", required=True, help="OpenAI API key.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for reproducibility.")
    parser.add_argument("--test-sample-num", type=int, default=300, help="Number of test samples to evaluate.")
    parser.add_argument("--max-retries", type=int, default=10, help="Number of tries before skipping the instance.")
    parser.add_argument("--output-folder", type=str, default="LLM_Results", help="The output folder's name to save the outputs.")
    parser.add_argument("--testing-sampling", type=int, default=0, help="A number assigned to the outputs as sampling.")
    return parser.parse_args()

def evaluate_dataset(client: OpenAI, args: argparse.Namespace, dataset: str, model_name: str, in_context: int, feature_num: int, config: str) -> None:
    file_name = create_file_name(args.output_folder,dataset, model_name, in_context, feature_num, config, testing_sampling=args.testing_sampling)
    print(f"Processing: {file_name}")

    names, x_incontext, x_test, y_incontext, y_test = read_dataset(dataset, config)
    existing_df = pd.read_csv(file_name) if os.path.exists(file_name) else pd.DataFrame()

    additional_instruction = get_additional_instruction(dataset, names[-1]) if in_context == 0 else ""
    explanation = create_explanation(names[-1], additional_instruction, "Reasoning" in config)

    messages = [{"role": "system", "content": explanation}]
    for x, y in zip(x_incontext[:in_context], y_incontext[:in_context]):
        ex_context, ex_output = create_IO_example(dataset, x, y, feature_num, names, config)
        messages.append({"role": "user", "content": ex_context})
        messages.append({"role": "assistant", "content": ex_output})

    for num, (x, y) in enumerate(zip(x_test[:args.test_sample_num], y_test[:args.test_sample_num])):
        if num < len(existing_df):
            continue
        test_context, _ = create_IO_example(dataset, x, y, feature_num, names, config)
        messages.append({"role": "user", "content": test_context})

        for _ in range(args.max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1000 if "Reasoning" in config else 10,
                    temperature=0.1,
                    seed=args.seed
                )
                response_text = response.choices[0].message.content
                processed_response = process_response(response_text, "Reasoning" in config)
                
                df = pd.DataFrame([{"raw_text": response_text, "processed_response": processed_response}])
                df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)

                messages.pop()
                break
            except Exception as e:
                print(f"Error processing response: {e}")
                args.seed += 1
                time.sleep(1)
        else:
            print(f"Failed to process response after {args.max_retries} attempts. Skipping and replacing the number with -1.")
            df = pd.DataFrame([{"raw_text": response_text, "processed_response": -1.0}])
            df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)

def main():
    args = parse_arguments()
    client = OpenAI(api_key=args.api_key)
    # Iterate through all combinations of datasets, models, in-context examples, features, and configurations
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
                        
                        evaluate_dataset(client, args, dataset, model_name, in_context, feature_num, config)

if __name__ == "__main__":
    main()