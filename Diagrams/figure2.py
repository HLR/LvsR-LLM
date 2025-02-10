import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Arc
from matplotlib.ticker import MultipleLocator
from Diagrams.utils import pre_process, preprocess_whole_picture, dataset_names, features, models, in_contexts


def process_dataset_wholecontext(df, dataset):
    """
    Process the dataset for whole context visualization.

    Args:
    df (pd.DataFrame): Input dataframe
    dataset (str): Name of the dataset to process

    Returns:
    pd.DataFrame: Processed dataframe
    """
    df = df[df['dataset'] == dataset]
    df = df[(df['features'].isin(features)) &
            (df['model'].isin(models)) &
            (df['in_context'].isin(in_contexts)) &
            (df['config'].isin(["Named_Features", "Anonymized_Features"]))]
    return df

def main_figure2(evaluation_results,admission_chance_results,insurance_cost_results,used_car_prices_results,metric,dpi):
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
    admission_chance_results = pd.read_csv(admission_chance_results)
    insurance_cost_results = pd.read_csv(insurance_cost_results)
    used_car_prices_results = pd.read_csv(used_car_prices_results)

    df_raw = pd.concat([evaluation_results, admission_chance_results,
                        insurance_cost_results, used_car_prices_results],
                       ignore_index=True)

    # Define Y_SIZE based on the metric
    Y_SIZE = {
        "r2": {"Admission_Chance": 0.55, "Insurance_Cost": 0.9, "Used_Car_Prices": 1.0},
        "MSE": {"Admission_Chance": 0.01, "Insurance_Cost": 10 ** 8 * 1.5, "Used_Car_Prices": 10 ** 9 * 2},
        "MAE": {"Admission_Chance": 0.08, "Insurance_Cost": 10 ** 4 * 1.0, "Used_Car_Prices": 10 ** 4 * 2.5}
    }.get(metric, {"Admission_Chance": 0.7, "Insurance_Cost": 0.9, "Used_Car_Prices": 1.0})

    scientific_formatter, formatter, features, in_contexts, models = pre_process(plt, 11)
    fig, axes, palette, main_angles, sub_angles, angles, labels = preprocess_whole_picture(plt, dpi=dpi)

    palette = [sns.color_palette("viridis")[0], sns.color_palette("viridis")[3], sns.color_palette("viridis")[4]]
    palette = palette + palette

    for i, dataset in enumerate(dataset_names):
        dataset_name = dataset
        axes[i].yaxis.set_major_locator(MultipleLocator(Y_SIZE[dataset] / 5))

        df = process_dataset_wholecontext(df_raw, dataset)

        # Draw circular arcs
        for j, color in enumerate(['black', 'grey', 'white']):
            arc = Arc((0, 0), 2 * Y_SIZE[dataset], 2 * Y_SIZE[dataset], theta1=j * 120 - 20, theta2=j * 120 + 120 - 20, color=color, linewidth=10, transform=axes[i].transData._b)
            axes[i].add_patch(arc)

        # Calculate angles for the radar chart
        main_angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False)
        sub_angles = np.linspace(0, 2 * np.pi / len(models), len(features), endpoint=False)
        angles = np.concatenate([main_angle + sub_angles for main_angle in main_angles])
        angles = np.concatenate((angles, [angles[0]]))

        # Create labels
        labels = [f'F{f}' for model in models for f in features]

        # Plot data for each configuration and in-context learning setting
        for config in ["Named_Features", "Anonymized_Features"]:
            for in_context, color in zip(df['in_context'].unique(), palette):
                values = []
                for model in models:
                    for feature in features:
                        metric_value = df[(df['config'] == config) &
                                          (df['features'] == feature) &
                                          (df['in_context'] == in_context) &
                                          (df['model'] == model)][metric].values
                        if metric == "r2":
                            metric_value = 1 - metric_value
                        values.append(metric_value[0] if len(metric_value) > 0 else np.nan)
                values = np.concatenate((values, [values[0]]))

                line_form = '-' if config == "Named_Features" else "--"
                axes[i].plot(angles, values, line_form, linewidth=1.5, label=f"{config.replace("_", " ")} &\n{in_context} In-Context Examples", color=color)
                axes[i].fill(angles, values, alpha=0.1, color=color)

        # Set up the axes
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels([])
        axes[i].set_ylim(0, Y_SIZE[dataset])
        for idx, f in enumerate(labels):
            axes[i].text(angles[idx], axes[i].get_ylim()[1] * 1.09, f, ha='center', va='center', size=12, rotation=-90 + np.degrees(angles[idx]))

        # Adjust label positioning
        for label, angle in zip(axes[i].get_xticklabels(), angles[:-1]):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        # Add model labels
        for idx, model in enumerate(['GPT-3', "LLaMA 3", 'GPT-4']):
            angle = main_angles[idx]
            axes[i].text(angle + np.pi / 3 - np.pi / 9, axes[i].get_ylim()[1] * 1.21, model,
                         ha='center', va='center', size=14, weight='bold',
                         rotation=-90 + np.degrees(angle + np.pi / 3 - np.pi / 9))

        # Format y-axis
        axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[i].yaxis.set_major_formatter(formatter)
        axes[i].set_title(f'{dataset.replace("_", " ")}', fontsize=18, y=1.10, fontweight="bold")

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(-0.12, 0.5), ncol=1, fontsize=16)
    plt.tight_layout()
    plt.show()
