import pandas as pd, numpy as np,seaborn as sns,warnings
from fontTools.config import Config
from matplotlib.ticker import ScalarFormatter


dataset_names = ["Admission_Chance", "Insurance_Cost", "Used_Car_Prices"]
features = [1, 2, 3]
in_contexts = [10, 30, 100]
models = ['GPT-3','LLaMA 3', 'GPT-4']
configs= ["Named_Features", "Anonymized_Features", "Randomized_Ground_Truth", "Direct QA"]

def pre_process(plt, global_font_size=9):
    """
    Set up global plot parameters and formatters.

    Args:
        plt: matplotlib.pyplot object
        global_font_size (int): Global font size for the plot

    Returns:
        tuple: Containing scientific formatter, scalar formatter, and other global variables
    """
    # Update global font size
    plt.rcParams.update({'font.size': global_font_size})

    # Define scientific notation formatter
    def scientific_formatter(x, pos):
        return f'{x:.2e}'

    # Set up scalar formatter for scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    return (scientific_formatter, formatter, features, in_contexts, models)


def preprocess_whole_picture(plt, dpi=300):
    """
    Set up the main figure and axes for the visualization.

    Args:
        plt: matplotlib.pyplot object
        dpi (int): Dots per inch for the figure

    Returns:
        tuple: Containing figure, axes, color palette, and angle calculations
    """
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=False, dpi=dpi, subplot_kw=dict(projection='polar'))

    # Define color palette
    palette = [sns.color_palette("viridis")[i] for i in [0, 3, 5]] * 2

    # Calculate angles for the polar plot
    main_angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False)
    sub_angles = np.linspace(0, 2 * np.pi / len(models), len(features) * len(in_contexts), endpoint=False)
    angles = np.concatenate([main_angle + sub_angles for main_angle in main_angles])
    angles = np.concatenate((angles, [angles[0]]))  # Close the circle

    # Create labels
    labels = [f'F{f}' for model in models for ic in in_contexts for f in features]

    return fig, axes, palette, main_angles, sub_angles, angles, labels


def process_dataset_whole_picture(df, dataset):
    """
    Process and filter the dataset for visualization.

    Args:
        df (pd.DataFrame): Input dataframe
        dataset (str): Dataset name to filter by

    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Filter dataframe based on dataset and other criteria
    df = df[(df['dataset'] == dataset) &
            (df['features'].isin(features)) &
            (df['model'].isin(models + ["Mean Model"])) &
            (df['in_context'].isin([0] + in_contexts)) &
            (df['config'].isin(configs + ["Real"]))]

    # Add rows for Direct QA with different in-context sizes
    df_direct_qa = df[df['config'] == "Direct QA"].copy()
    for ic in [10, 30, 100]:
        df_direct_qa_copy = df_direct_qa.copy()
        df_direct_qa_copy['in_context'] = ic
        df = pd.concat([df, df_direct_qa_copy])

    # Add rows for Mean Model with different model names
    df_mean_model = df[df['model'] == "Mean Model"].copy()
    for model in models:
        df_mean_model_copy = df_mean_model.copy()
        df_mean_model_copy['config'] = "Mean Model"
        df_mean_model_copy['model'] = model
        df = pd.concat([df, df_mean_model_copy])

    return df


def axis_post_process_whole_picture(axes, config, angles, color, labels, dataset, Y_SIZE, main_angles, in_contexts,
                                    models, values):
    """
    Post-process the polar plot axis with labels, ticks, and additional text.

    Args:
        axes: matplotlib axes object
        config (str): Configuration name
        angles (np.array): Angles for the polar plot
        color: Color for the plot line
        labels (list): Labels for x-ticks
        dataset (str): Dataset name
        Y_SIZE (function): Function to determine y-axis limit
        main_angles (np.array): Main angles for model labels
        in_contexts (list): In-context learning sizes
        models (list): Model names
        values (np.array): Values to plot
    """
    values = np.concatenate((values, [values[0]]))  # Close the circle

    # Plot the line based on the configuration
    if config == "Direct QA":
        axes.plot(angles, values, '-.', linewidth=1.5, label=config.replace("_", " "), color="black")
    elif config == "Mean Model":
        axes.plot(angles, values, '--', linewidth=1.5, label=config.replace("_", " "), color="red")
    else:
        axes.plot(angles, values, '-', linewidth=1.5, label=config.replace("_", " "), color=color)
        if config not in ["Direct QA", "Mean Model"]:
            axes.fill(angles, values, alpha=0.1, color=color)

    # Set x-ticks and labels
    axes.set_xticks(angles[:-1])
    axes.set_xticklabels([])
    axes.set_ylim(0, Y_SIZE(dataset))
    for idx, f in enumerate(labels):
        axes.text(angles[idx], axes.get_ylim()[1] * 1.11, f, ha='center', va='center', size=10,rotation=-90 + np.degrees(angles[idx]))


    # Add model and in-context labels
    for idx, model in enumerate(models):
        angle = main_angles[idx]
        axes.text(angle + np.pi / 3, axes.get_ylim()[1] * 1.31, model,
                  ha='center', va='center', size=14, weight='bold', rotation=-90 + np.degrees(angle + np.pi / 3))

        for jdx, ic in enumerate(in_contexts):
            sub_angle = angle + (jdx + 0.3) * (2 * np.pi / (len(models) * len(in_contexts)))
            axes.text(sub_angle, axes.get_ylim()[1] * 1.21, f'IC{ic}', ha='center', va='center', size=12,rotation=-90 + np.degrees(sub_angle))

def preprocess_context(plt,dpi=300):
    """
    Prepare the matplotlib figure and axes for visualization.

    Args:
        plt: matplotlib pyplot object
        dpi (int): Dots per inch for the figure

    Returns:
        tuple: Figure, axes, and color palette
    """
    fig, axes = plt.subplots(1, 3, figsize=(8, 5), sharey=False,dpi=dpi)
    colors = sns.color_palette("mako")
    colors = [sns.color_palette("viridis")[0],sns.color_palette("viridis")[3],sns.color_palette("viridis")[5]]+["black","grey"]
    return fig, axes, colors


def process_dataset_context(dataset_df,dataset,feature_num=3):
    """
        Filter and process the dataset for context-based analysis.

        Args:
            dataset_df (pd.DataFrame): Input dataframe
            dataset (str): Name of the dataset to filter
            feature_num (int): Number of features to consider

        Returns:
            pd.DataFrame: Processed dataframe
        """
    dataset_df= dataset_df[dataset_df['dataset'] == dataset]
    dataset_df = pd.concat([
        dataset_df[
            (dataset_df['features'].isin([feature_num])) &
            (dataset_df['model'].isin(['GPT-3', "LLaMA 3", 'GPT-4'])) &
            (dataset_df['in_context'].isin([10,30,100])) &
            (dataset_df['config'].isin(['Named_Features','Anonymized_Features']))
        ],
        dataset_df[
            (dataset_df['features'].isin([feature_num])) &
            (dataset_df['model'].isin(["Ridge", "RandomForest"])) &
            (dataset_df['in_context'].isin([10,30,100])) &
            (dataset_df['config'].isin(['Real']))
        ]
    ], ignore_index=True)
    return dataset_df

def process_dataset_feature(dataset_df,dataset,in_context_num=100):
    """
    Filter and process the dataset for feature-based analysis.

    Args:
        dataset_df (pd.DataFrame): Input dataframe
        dataset (str): Name of the dataset to filter
        in_context_num (int): In-context learning size to consider

    Returns:
        pd.DataFrame: Processed dataframe
    """
    dataset_df = dataset_df[dataset_df['dataset'] == dataset]
    dataset_df = pd.concat([
        dataset_df[
            (dataset_df['features'].isin([1,2,3])) &
            (dataset_df['model'].isin(['GPT-3', "LLaMA 3", 'GPT-4'])) &
            (dataset_df['in_context'].isin([in_context_num])) &
            (dataset_df['config'].isin(['Named_Features','Anonymized_Features']))
        ],
        dataset_df[
            (dataset_df['features'].isin([1,2,3])) &
            (dataset_df['model'].isin(["Ridge", "RandomForest"])) &
            (dataset_df['in_context'].isin([in_context_num])) &
            (dataset_df['config'].isin(['Real']))
        ]
    ], ignore_index=True)
    return dataset_df



def axis_post_process_context(axes,x,width,formatter,dataset,Y_SIZE,x_text=[10,30,100]):
    """
    Post-process the axis for context-based visualization.

    Args:
        axes: matplotlib axes object
        x: x-coordinates for the bars
        width: width of the bars
        formatter: formatter for y-axis labels
        dataset (str): Name of the dataset
        Y_SIZE (function): Function to determine y-axis limit
        x_text (list): Labels for x-axis ticks
    """
    axes.set_title(f'{dataset.replace("_"," ")}', fontsize=16,y=1.06,fontweight ="bold")
    axes.set_xticks(x + width)
    axes.set_xticklabels(x_text)
    axes.set_ylim(0, Y_SIZE(dataset))
    axes.tick_params(axis='x')
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes.yaxis.set_major_formatter(formatter)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)