import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from Diagrams.utils import pre_process, preprocess_whole_picture, process_dataset_whole_picture, axis_post_process_whole_picture, dataset_names, configs
import seaborn as sns

def main_figure1(evaluation_results,admission_chance_results,insurance_cost_results,used_car_prices_results,metric,dpi):
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
    evaluation_results = pd.read_csv(evaluation_results)
    Admission_Chance_MLResults = pd.read_csv(admission_chance_results)
    Insurance_Cost_MLResults = pd.read_csv(insurance_cost_results)
    Used_Car_Prices_MLResults = pd.read_csv(used_car_prices_results)

    df_raw = pd.concat([evaluation_results, Admission_Chance_MLResults, Insurance_Cost_MLResults, Used_Car_Prices_MLResults], ignore_index=True)

    # Define Y_SIZE based on the metric
    Y_SIZE = {
        "Admission_Chance": 1.2,
        "Insurance_Cost": 1.2,
        "Used_Car_Prices": 1.5
    }

    if metric == "MSE":
        Y_SIZE = {
            "Admission_Chance": 0.025,
            "Insurance_Cost": 10**9/5*1.2,
            "Used_Car_Prices": 10**9*3
        }
    elif metric == "MAE":
        Y_SIZE = {
            "Admission_Chance": 0.15,
            "Insurance_Cost": 10**4*1.2,
            "Used_Car_Prices": 10**4*5
        }

    Y_SIZE = Y_SIZE.get  # Convert to function for later use

    # Prepare plot elements
    scientific_formatter, formatter, features, in_contexts, models = pre_process(plt, 9)
    fig, axes, palette, main_angles, sub_angles, angles, labels = preprocess_whole_picture(plt, dpi=dpi)

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
                        metric_value = df[(df['config'] == config) & (df['features'] == feature) & (df['in_context'] == in_context) & (df['model'] == model)][metric].values
                        if metric == "r2":
                            metric_value = 1 - metric_value
                        values.append(metric_value[0] if len(metric_value) > 0 else np.nan)

            axis_post_process_whole_picture(axes[i], config, angles, color, labels, dataset, Y_SIZE, main_angles, in_contexts, models, values)

        # Set axis formatting
        axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[i].yaxis.set_major_formatter(formatter)
        offset = axes[i].yaxis.get_offset_text()
        offset.set_position((-0.1, 1.05))
        axes[i].set_title(f'{dataset.replace("_", " ")}', fontsize=16, y=1.2, fontweight="bold")

    # Add legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.23), ncol=5, fontsize=16)

    plt.tight_layout()
    plt.show()


