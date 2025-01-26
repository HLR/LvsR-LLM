import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from Diagrams.utils import models, dataset_names

def main_figure3(evaluation_results, admission_chance_results, insurance_cost_results, used_car_prices_results, metric, dpi):
    """
    Generate and display a complex figure comparing different models and datasets.

    Args:
        evaluation_results (str): Path to the evaluation results CSV file.
        admission_chance_results (str): Path to the admission chance results CSV file.
        insurance_cost_results (str): Path to the insurance cost results CSV file.
        used_car_prices_results (str): Path to the used car prices results CSV file.
        metric (str): The metric to use for comparison ('r2', 'MSE', or 'MAE').
        dpi (int): The resolution of the output figure.
    """
    # Load and combine data
    df_raw = pd.concat([
        pd.read_csv(file) for file in [
            evaluation_results,
            admission_chance_results,
            insurance_cost_results,
            used_car_prices_results
        ]
    ], ignore_index=True)

    # Define Y-axis limits for different metrics and datasets
    Y_SIZE = {
        "r2": {"Admission_Chance": 1.8, "Insurance_Cost": 1.2, "Used_Car_Prices": 1.5},
        "MSE": {"Admission_Chance": 0.025, "Insurance_Cost": 10**9/5*1.2, "Used_Car_Prices": 10**9*3},
        "MAE": {"Admission_Chance": 0.16, "Insurance_Cost": 10**4*1.3, "Used_Car_Prices": 10**4*4}
    }.get(metric, {})

    # Set up the plot
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 3, figsize=(9, 5), sharey=False, dpi=dpi)

    # Filter data and set up formatting
    filtered_df = df_raw[(df_raw['in_context'] == 0) | (df_raw['model'] == "Mean Model")]
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    # Define color palette
    colors = sns.color_palette("mako")

    # Plot for each dataset
    for i, dataset in enumerate(dataset_names):
        ax = axes[i]
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        dataset_df = filtered_df[filtered_df['dataset'] == dataset]

        x = np.arange(3)
        width = 0.2

        # Plot bars for each model
        for j, model in enumerate(models):
            model_data = dataset_df[dataset_df['model'] == model]
            if not model_data.empty:
                values = [model_data[model_data['features'] == fn][metric].values[0] for fn in [1,2,3]]
                if metric == "r2":
                    values = [1-v for v in values]
                ax.bar(x + j*width, values, width, label=model, color=colors[j*2+1])

        # Set labels and titles
        if i == 0:
            ax.set_ylabel("R2" if metric == "r2" else metric, fontsize=14)
        ax.set_title(f'{dataset.replace("_", " ")}', fontsize=16, y=1.06, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([1,2,3])
        if i == 1:
            ax.set_xlabel('Number of Features', fontsize=14)

        # Set y-axis limits and format
        y_limit = Y_SIZE.get(dataset)
        if y_limit:
            ax.set_ylim(0, y_limit)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.set_major_formatter(formatter)

        # Add mean line
        mean_value = dataset_df[dataset_df["model"] == "Mean Model"][metric].mean()
        if metric == "r2":
            mean_value = 1 - mean_value
        ax.axhline(y=mean_value, color='r', linestyle='--', linewidth=1)

        # Additional formatting for specific cases
        # if dataset == "Admission_Chance" and metric == "MSE":
        #     ax.yaxis.set_major_locator(MultipleLocator(0.005))
        ax.tick_params(axis='x')

    # Add legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=11)

    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.9)
    plt.show()