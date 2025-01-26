import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Diagrams.utils import pre_process, preprocess_context, process_dataset_feature, axis_post_process_context, \
    dataset_names


def main_figure4(evaluation_results, admission_chance_results, insurance_cost_results, used_car_prices_results, metric,
                 dpi, in_context_num=100):
    """
    Generate and display a complex figure comparing different models and datasets.

    Args:
        evaluation_results (str): Path to the evaluation results CSV file.
        admission_chance_results (str): Path to the admission chance results CSV file.
        insurance_cost_results (str): Path to the insurance cost results CSV file.
        used_car_prices_results (str): Path to the used car prices results CSV file.
        metric (str): The metric to use for comparison ('r2', 'MSE', or 'MAE').
        dpi (int): The resolution of the output figure.
        in_context_num (int, optional): The number of in-context examples. Defaults to 100.
    """

    # Load and concatenate all datasets
    df_raw = pd.concat([
        pd.read_csv(file) for file in [
            evaluation_results,
            admission_chance_results,
            insurance_cost_results,
            used_car_prices_results
        ]
    ], ignore_index=True)

    # Define Y-axis scale for different metrics and datasets
    Y_SIZE = {
        "r2": {"Admission_Chance": 0.4, "Insurance_Cost": 0.8, "Used_Car_Prices": 1.0}.get,
        "MSE": {"Admission_Chance": 0.008, "Insurance_Cost": 10 ** 8 * 1.2, "Used_Car_Prices": 10 ** 9 * 1.8}.get,
        "MAE": {"Admission_Chance": 0.07, "Insurance_Cost": 10 ** 4 * 0.8, "Used_Car_Prices": 10 ** 4 * 3}.get
    }.get(metric, lambda x: None)

    # Prepare plot formatting and data
    scientific_formatter, formatter, features, _, models = pre_process(plt, 12)
    fig, axes, palette = preprocess_context(plt, dpi=dpi)

    # Plot data for each dataset
    for i, dataset in enumerate(dataset_names):
        dataset_df = process_dataset_feature(df_raw.copy(), dataset, in_context_num)
        x, width = np.arange(len(features)), 0.2

        # Plot data for each model
        for j, model in enumerate(models + ["Ridge", "RandomForest"]):
            model_data = dataset_df[dataset_df['model'] == model]

            if model in ["Ridge", "RandomForest"]:
                values = [model_data[model_data['features'] == f][metric].values[0] for f in [1, 2, 3]]
                if metric == "r2":
                    values = [1 - v for v in values]
                axes[i].plot(x + width, values, linestyle="--", label=model, color=palette[j])
            else:
                for config in ['Anonymized_Features', 'Named_Features']:
                    values = [
                        model_data[(model_data['features'] == f) & (model_data['config'] == config)][metric].values[0]
                        for f in [1, 2, 3]]
                    if metric == "r2":
                        values = [1 - v for v in values]
                    axes[i].plot(x + width, values,
                                 linestyle='-' if config == "Named_Features" else '-.',
                                 label=f"{model} & {config.replace('_', ' ')}",
                                 color=palette[j])

        # Set labels and format axes
        axes[0].set_ylabel("1 - R2" if metric == "r2" else metric, fontsize=14)
        axes[1].set_xlabel('Number of Features', fontsize=14)
        axis_post_process_context(axes[i], x, width, formatter, dataset, Y_SIZE, [1, 2, 3])

    # Add legend
    handles, labels = axes[1].get_legend_handles_labels()
    handles = [handles[i * 2] for i in range(3)] + [handles[i * 2 + 1] for i in range(3)] + [handles[-2], handles[-1]]
    labels = [labels[i * 2] for i in range(3)] + [labels[i * 2 + 1] for i in range(3)] + [labels[-2], labels[-1]]
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=11)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.9)
    plt.show()