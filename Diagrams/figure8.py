import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Diagrams.utils import pre_process, preprocess_context, process_dataset_feature, axis_post_process_context, dataset_names

def main_figure8(evaluation_results, admission_chance_results, insurance_cost_results, used_car_prices_results, metric, dpi):
    scientific_formatter, formatter, features, in_contexts, models = pre_process(plt,12)
    Y_SIZE = {"ChanceOfAdmition":0.04, "insurance":10**8*2, "usedcars":10**9*3}.get
    df_raw = pd.concat([
        pd.read_csv(file) for file in [
            evaluation_results,
            admission_chance_results,
            insurance_cost_results,
            used_car_prices_results
        ]
    ], ignore_index=True)

    colors = sns.color_palette("mako")
    selected_colors = [colors[1], colors[3], colors[5]]  # Using 2nd, 4th and 6th colors

    fig, axs = plt.subplots(3,3, figsize=(12, 18))
    for idy, model in enumerate(models):
        for idx, dataset in enumerate(dataset_names):
            axes = axs[idy][idx]
            filtered_df = df_raw[(df_raw['model'] == model) & (df_raw['dataset'] == dataset) & (df_raw['features'].isin([1,2,3]))& (df_raw['in_context'] == 0) & (df_raw['config'].isin(["Direct QA","Reasoning"]))]
            pivot_table = filtered_df.pivot_table(index='config', columns='features', values='MSE', aggfunc='mean')
            pivot_table.plot(kind='bar', ax=axes, width=0.8, color=selected_colors)
            axes.set_ylim(0, Y_SIZE(dataset))
            axes.set_xlabel(None)
            if idx==0:
                axes.set_ylabel('MSE', fontsize=14)
            axes.legend_.remove()
            axes.grid(False)
            if idy==0:
                axes.set_title(f'{dataset.replace("_", " ")}', fontsize=20, y=1.1, fontweight="bold")

            axes.set_xticklabels(["Direct QA","Reasoning"], rotation=0)
            axes.set_ylim(0, Y_SIZE(dataset))
            axes.spines['right'].set_visible(False)
            axes.spines['top'].set_visible(False)

    for idx, model in enumerate(models):
        fig.text(-0.01, 0.18 + (idx * 0.32), f'{model}', ha='center', va='center', rotation=90, fontsize=17, fontweight="bold")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, ["1 Feature", "2 Features", "3 Features"], loc='center', bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=14)
    plt.tight_layout()
    plt.show()