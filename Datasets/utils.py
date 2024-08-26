import numpy as np

def generate_random_output(dataset):
    if dataset == "Admission_Chance":
        return generate_random_chance_of_admission()
    elif dataset == "Used_Car_Prices":
        return generate_random_charges_usedcar()
    elif dataset == "Insurance_Cost":
        return generate_random_charges_insurance()

def generate_random_chance_of_admission(mean=0.72435, std=0.14260933017384092):
    return np.random.normal(loc=mean, scale=std)

def generate_random_charges_usedcar(mean=50014.51, std=42279.49):
    return np.random.normal(loc=mean, scale=std)

def generate_random_charges_insurance(mean=13270.422265141257, std=12110.011236694001):
    return np.random.normal(loc=mean, scale=std)

def get_mean_prediction(dataset):
    if dataset == "Admission_Chance":
        return 0.72
    elif dataset == "Used_Car_Prices":
        return 50014.51
    elif dataset == "Insurance_Cost":
        return 13270.42