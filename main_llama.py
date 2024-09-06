import argparse
import os
import time

import pandas as pd
import replicate

from utils import create_file_name, create_IO_example, get_additional_instruction, create_explanation, process_response
from reader import read_dataset

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Dataset Evaluation Script for Meta Models.")
    parser.add_argument("--datasets", nargs="+", default=["Insurance_Cost", "Admission_Chance", "Used_Car_Prices"], help="List of datasets to evaluate.")
    parser.add_argument("--models", nargs="+", default=["meta/meta-llama-3-70b-instruct"], help="List of Replicate models to use.")
    parser.add_argument("--in-context-numbers", nargs="+", type=int, default=[0, 10, 30, 100], help="List of in-context example numbers to use.")
    parser.add_argument("--feature-nums", nargs="+", type=int, default=[1, 2, 3], help="List of feature numbers to use.")
    parser.add_argument("--configs", nargs="+", default=["Named_Features", "Anonymized_Features", "Randomized_Ground_Truth", "Reasoning"], choices=["Named_Features", "Anonymized_Features", "Randomized_Ground_Truth", "Reasoning", "Missing_Inputs", "Missing_Inputs_and_Anonymized_Features"], help="List of prompt configurations to use.")    
    parser.add_argument("--api-key-token", required=True, help="Replicate AI API token.")
    parser.add_argument("--test-sample-num", type=int, default=300, help="Number of test samples to evaluate.")
    parser.add_argument("--max-retries", type=int, default=10, help="Number of tries before skipping the instance.")
    parser.add_argument("--output-folder", type=str, default="LLM_Results", help="The output folder's name to save the outputs.")
    parser.add_argument("--testing-sampling", type=int, default=0, help="A number assigned to the outputs as sampling.")
    return parser.parse_args()

def run_replicate_model(model_name: str, prompt: str, max_tokens: int) -> str:
    response = replicate.run(
        model_name,
        input={
            "top_p": 0.99,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "min_tokens": 1,
            "temperature": 0.1,
            "prompt_template": "{prompt}"
        },
    )
    return "".join(response)

def evaluate_dataset(args: argparse.Namespace, dataset: str, model_name: str, in_context: int, feature_num: int, config: str) -> None:
    file_name = create_file_name(args.output_folder,dataset, model_name, in_context, feature_num, config, testing_sampling=args.testing_sampling)
    print(f"Processing: {file_name}")

    names, x_incontext, x_test, y_incontext, y_test = read_dataset(dataset, config)
    existing_df = pd.read_csv(file_name) if os.path.exists(file_name) else pd.DataFrame()

    additional_instruction = get_additional_instruction(dataset, names[-1]) if in_context == 0 else ""
    explanation = create_explanation(names[-1], additional_instruction, ("Named_Features" in config) or ("Reasoning" in config))
    
    prompt_messages = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{explanation}<|eot_id|>"
    
    for x, y in zip(x_incontext[:in_context], y_incontext[:in_context]):
        ex_context, ex_output = create_IO_example(dataset, x, y, feature_num, names, config)
        prompt_messages += f"<|start_header_id|>user<|end_header_id|>\n\n{ex_context}<|eot_id|>"
        prompt_messages += f"<|start_header_id|>assistant<|end_header_id|>\n\n{ex_output}<|eot_id|>"

    failures = 0
    for num, (x, y) in enumerate(zip(x_test[:args.test_sample_num], y_test[:args.test_sample_num])):
        print(num, end=" ")
        if num < len(existing_df):
            continue

        ex_context, _ = create_IO_example(dataset, x, y, feature_num, names, config)
        cur_prompt_messages = prompt_messages + f"<|start_header_id|>user<|end_header_id|>\n\n{ex_context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        for _ in range(args.max_retries):
            try:
                if "Reasoning" in config:
                    response_text = run_replicate_model(model_name, cur_prompt_messages, 1000)
                else:
                    response_text = run_replicate_model(model_name, cur_prompt_messages, 6)
                processed_text = process_response(response_text, "Reasoning" in config)
                
                df = pd.DataFrame([{"raw_text": response_text, "processed_response": processed_text}])
                df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
                break
            except Exception as e:
                print(f"Error processing response: {e}")
                failures += 1
                if failures > args.max_failures:
                    print(f"Exceeded maximum number of failures ({args.max_failures}). Exiting.")
                    return
                time.sleep(20)
        else:
            print(f"Failed to process response after {args.max_retries} attempts. Skipping and replacing the number with -1.")
            df = pd.DataFrame([{"raw_text": response_text, "processed_response": -1.0}])
            df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
        
        time.sleep(1)

def main():
    args = parse_arguments()
    os.environ["REPLICATE_API_TOKEN"] = args.api_key_token
    # Iterate through all combinations of datasets, models, in-context examples, features, and configurations
    for dataset in args.datasets:
        for model_name in args.models:
            for in_context in args.in_context_numbers:
                for feature_num in args.feature_nums:
                    for config in args.configs:
                        # Skip certain combinations based on experimental constraints
                        if (in_context == 0 and not (("Named_Features" in config) or ("Reasoning" in config))) or \
                           (config == "Reasoning" and in_context > 0) or \
                           (dataset == "Admission_Chance" and in_context > 101) or \
                           (feature_num == 4 and dataset != "Used_Car_Prices"):
                            continue
                        
                        evaluate_dataset(args, dataset, model_name, in_context, feature_num, config)

if __name__ == "__main__":
    main()
