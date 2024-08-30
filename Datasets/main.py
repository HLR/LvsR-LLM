import argparse
from data_preprocessing import shuffle_and_split_data
from model_training import train_and_evaluate_models

def main():
    parser = argparse.ArgumentParser(description="Process and analyze datasets.")
    parser.add_argument("--action", choices=["shuffle", "train"], required=True, help="Action to perform.")
    parser.add_argument("--dataset", choices=["Insurance_Cost", "Admission_Chance", "Used_Car_Prices"], required=True, help="Dataset to process.")
    parser.add_argument("--output", help="Output CSV file for the ML results data.")
    args = parser.parse_args()

    if args.action == "shuffle": shuffle_and_split_data(args.dataset)
    elif args.action == "train": train_and_evaluate_models(args.dataset,args.output)

if __name__ == "__main__":
    main()