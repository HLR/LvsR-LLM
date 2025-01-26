import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Diagrams.utils import pre_process, preprocess_context, process_dataset_context, axis_post_process_context, \
    dataset_names


def main_figure3(evaluation_results, admission_chance_results, insurance_cost_results, used_car_prices_results, metric,
                 dpi, feature_num=3):
    """
    Generate and display a complex figure comparing different models and datasets.

    Args:
        evaluation_results (str): Path to the evaluation results CSV file.
        admission_chance_results (str): Path to the admission chance results CSV file.
        insurance_cost_results (str): Path to the insurance cost results CSV file.
        used_car_prices_results (str): Path to the used car prices results CSV file.
        metric (str): The metric to use for comparison ('r2', 'MSE', or 'MAE').
        dpi (int): The resolution of the output figure.
        feature_num (int, optional): The number of features. Defaults to 3.
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

    # Define Y-axis scale for different datasets and metrics
    Y_SIZE = {
        "r2": {"Admission_Chance": 0.6, "Insurance_Cost": 1.2, "Used_Car_Prices": 1.2}.get,
        "MSE": {"Admission_Chance": 0.012, "Insurance_Cost": 10 ** 9 / 5 * 0.80, "Used_Car_Prices": 10 ** 9 * 2}.get,
        "MAE": {"Admission_Chance": 0.10, "Insurance_Cost": 10 ** 4 * 0.9, "Used_Car_Prices": 10 ** 4 * 2.5}.get
    }

    # Prepare plot settings
    scientific_formatter, formatter, _, in_contexts, models = pre_process(plt, 12)
    fig, axes, palette = preprocess_context(plt, dpi=dpi)

    # Iterate through datasets
    for i, dataset in enumerate(dataset_names):
        dataset_df = process_dataset_context(df_raw.copy(), dataset, feature_num)
        x, width = np.arange(len(in_contexts)), 0.2

        # Plot data for each model
        for j, model in enumerate(models + ["Ridge", "RandomForest"]):
            model_data = dataset_df[dataset_df['model'] == model]

            if model in ["Ridge", "RandomForest"]:
                values = [model_data[model_data['in_context'] == fn][metric].values[0] for fn in [10, 30, 100]]
                if metric == "r2":
                    values = [1 - v for v in values]
                axes[i].plot(x + width, values, linestyle="--", label=model, color=palette[j])
            else:
                for config in ['Named_Features', 'Anonymized_Features']:
                    values = [
                        model_data[(model_data['in_context'] == fn) & (model_data['config'] == config)][metric].values[
                            0]
                        for fn in [10, 30, 100]
                    ]
                    if metric == "r2":
                        values = [1 - v for v in values]
                    axes[i].plot(x + width, values,
                                 linestyle='-' if config == "Named_Features" else '-.',
                                 label=f"{model} & {config.replace('_', ' ')}",
                                 color=palette[j])

        # Set labels and adjust axis
        axes[0].set_ylabel("R2" if metric == "r2" else metric, fontsize=14)
        axes[1].set_xlabel('Number of Examples', fontsize=14)
        axis_post_process_context(axes[i], x, width, formatter, dataset, Y_SIZE[metric])

    # Create legend
    handles, labels = axes[1].get_legend_handles_labels()
    handles = [handles[i * 2] for i in range(3)] + [handles[i * 2 + 1] for i in range(3)] + [handles[-2], handles[-1]]
    labels = [labels[i * 2] for i in range(3)] + [labels[i * 2 + 1] for i in range(3)] + [labels[-2], labels[-1]]
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=11)

    # Adjust layout and display plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.9)
    plt.show()