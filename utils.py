import numpy as np
import random
from typing import List, Tuple, Union

def create_file_name(output_folder: str, dataset: str, model_name: str, in_context: int, 
                     feature_num: int, configs: str, testing_sampling: str) -> str:
    """
    Create a standardized file name based on input parameters.

    Args:
        output_folder (str): The folder where the file will be saved.
        dataset (str): Name of the dataset.
        model_name (str): Name of the model.
        in_context (int): Number of in-context examples.
        feature_num (int): Number of features.
        configs (str): Prompt Configuration.
        testing_sampling (str): Testing sampling number.

    Returns:
        str: The generated file name.
    """
    model_name = model_name.split("/")[0]
    filename = f"{output_folder}/{dataset}_{model_name}_{in_context}_{feature_num}_{configs}_{testing_sampling}.csv"
    return filename

def generate_random_output(dataset: str) -> float:
    """
    Generate a random output based on the dataset.

    Args:
        dataset (str): Name of the dataset.

    Returns:
        float: A randomly generated output value.
    """
    params = {
        "insurance": (13270.422265141257, 12110.011236694001),
        "ChanceOfAdmition": (0.72435, 0.14260933017384092),
        "usedcars": (50014.51, 42279.49)
    }
    mean, std = params.get(dataset, (0, 1))
    return float(np.random.normal(loc=mean, scale=std))

def create_IO_example(dataset: str, x: List[Union[float, int, bool]], y: float, feature_num: int, Names: List[str], config: str, is_test: bool = False) -> Tuple[str, str]:
    """
    Create an input-output example for a given dataset and configuration.

    Args:
        dataset (str): Name of the dataset.
        x (List[Union[float, int]]): List of feature values.
        y (float): Target value.
        feature_num (int): Number of features.
        Names (List[str]): List of feature names.
        config (str): Configuration setting.
        is_test (bool, optional): Whether this is a test example. Defaults to False.

    Returns:
        Tuple[str, str]: Input context and output value as strings.
    """
    x_context = ""
    for i in range(feature_num):
        feature_name = Names[i]
        feature_value = "?" if "Missing_Inputs" in config and random.random() > 0.5 else str(x[i])
        x_context += f"{feature_name}: {feature_value}\n"

    output_value = generate_random_output(dataset) if config == "Randomized_Ground_Truth" and not is_test else y
    return x_context + f"{Names[-1]}: ", f"{output_value}\n\n"

def get_additional_instruction(dataset: str, target_name: str) -> str:
    """
    Get additional instructions based on the dataset.

    Args:
        dataset (str): Name of the dataset.
        target_name (str): Name of the target variable.

    Returns:
        str: Additional instruction string.
    """
    instructions = {
        "ChanceOfAdmition": f"The average of {target_name} is 0.74 with a standard deviation of 0.14.",
        "insurance": f"The average of {target_name} is 13270.42 with a standard deviation of 12110.01.",
        "usedcars": f"The average of {target_name} is 50014.51 with a standard deviation of 42279.49."
    }
    return instructions.get(dataset, "")

def create_explanation(target_name: str, additional_instruction: str, is_reasoning: bool) -> str:
    """
    Create an explanation string based on the target and whether it involves reasoning.

    Args:
        target_name (str): Name of the target variable.
        additional_instruction (str): Additional instructions to include.
        is_reasoning (bool): Whether this involves reasoning.

    Returns:
        str: The created explanation string.
    """
    base_explanation = f"The task is to provide your best number estimation for the {target_name}. "
    if is_reasoning:
        return (f"{base_explanation}Please explain your reasoning based on the given information and provide your "
                f"final estimation as just one number (not a range) in the last sentence of your explanation as such: "
                f"My final estimation is #. {additional_instruction}")
    else:
        return f"{base_explanation}Please provide just one number, without any additional text or explanation. {additional_instruction}"

def process_response(response_text: str, is_reasoning: bool) -> float:
    """
    Process the response text to extract the numerical estimation.

    Args:
        response_text (str): The response text to process.
        is_reasoning (bool): Whether the response includes reasoning.

    Returns:
        float: The extracted numerical estimation.
    """
    if not is_reasoning:
        response_text = response_text.split(":")[-1].split("is")[-1].strip("$. %").replace(",", "")
    return float(response_text)