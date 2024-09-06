import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from utils import pre_process, preprocess_whole_picture, process_dataset_whole_picture, axis_post_process_whole_picture, dataset_names, configs

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize ML evaluation results.")
    parser.add_argument("--evaluation_results", default="../evaluation_results.csv", help="Path to evaluation results CSV")
    parser.add_argument("--admission_chance", default="../Datasets/Admission_Chance_MLResults.csv", help="Path to Admission Chance ML results CSV")
    parser.add_argument("--insurance_cost", default="../Datasets/Insurance_Cost_MLResults.csv", help="Path to Insurance Cost ML results CSV")
    parser.add_argument("--used_car_prices", default="../Datasets/Used_Car_Prices_MLResults.csv", help="Path to Used Car Prices ML results CSV")
    parser.add_argument("--output", default="./Figure4.png", help="Output figure path and name")
    parser.add_argument("--dpi", type=int, default=300, help="DPI of the output figure")
    parser.add_argument("--metric", default="MSE", choices=["MSE", "MAE", "r2"], help="Input metric for evaluation")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load data
    evaluation_results = pd.read_csv(args.evaluation_results)
    Admission_Chance_MLResults = pd.read_csv(args.admission_chance)
    Insurance_Cost_MLResults = pd.read_csv(args.insurance_cost)
    Used_Car_Prices_MLResults = pd.read_csv(args.used_car_prices)

    df_raw = pd.concat([evaluation_results, Admission_Chance_MLResults, Insurance_Cost_MLResults, Used_Car_Prices_MLResults], ignore_index=True)

    # Define Y_SIZE based on the metric
    Y_SIZE = {
        "Admission_Chance": 1.2,
        "Insurance_Cost": 1.2,
        "Used_Car_Prices": 1.5
    }

    if args.metric == "MSE":
        Y_SIZE = {
            "Admission_Chance": 0.025,
            "Insurance_Cost": 10**9/5*1.2,
            "Used_Car_Prices": 10**9*3
        }
    elif args.metric == "MAE":
        Y_SIZE = {
            "Admission_Chance": 0.15,
            "Insurance_Cost": 10**4*1.2,
            "Used_Car_Prices": 10**4*5
        }

    Y_SIZE = Y_SIZE.get  # Convert to function for later use

    # Prepare plot elements
    scientific_formatter, formatter, features, in_contexts, models = pre_process(plt, 9)
    fig, axes, palette, main_angles, sub_angles, angles, labels = preprocess_whole_picture(plt, dpi=args.dpi)

    # Process and plot data for each dataset
    for i, dataset in enumerate(dataset_names):
        df = process_dataset_whole_picture(df_raw[:], dataset)

        # Draw arcs
        for j, color in enumerate(['black', 'grey', 'white']):
            arc = Arc((0, 0), 2 * Y_SIZE(dataset), 2 * Y_SIZE(dataset), theta1=j * 120, theta2=j * 120 + 120, color=color, linewidth=10, transform=axes[i].transData._b)
            axes[i].add_patch(arc)

        # Plot data for each configuration
        for config, color in zip(configs + ["Mean Model"], palette):
            values = []
            for model in models:
                for in_context in in_contexts:
                    for feature in features:
                        metric_value = df[(df['config'] == config) & (df['features'] == feature) & (df['in_context'] == in_context) & (df['model'] == model)][args.metric].values
                        if args.metric == "r2":
                            metric_value = 1 - metric_value
                        values.append(metric_value[0] if len(metric_value) > 0 else np.nan)

            axis_post_process_whole_picture(axes[i], config, angles, color, labels, dataset, Y_SIZE, main_angles, in_contexts, models, values)

        # Set axis formatting
        axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[i].yaxis.set_major_formatter(formatter)
        axes[i].set_title(f'{dataset.replace("_", " ")}', fontsize=16, y=1.15, fontweight="bold")

    # Add legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.23), ncol=5, fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    plt.show()

if __name__ == "__main__":
    main()

