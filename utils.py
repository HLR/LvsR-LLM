import numpy as np
import random

def create_file_name(dataset,model_name,sample_num,feature_num,configs,testing_sampling):
    filename="newoutput/"
    if "/" in model_name:
        model_name=model_name.split("/")[0]
    filename+=str(dataset)+"_"+str(model_name)+"_"+str(sample_num)+"_"+str(feature_num)
    filename+="_"+str(configs)+"_"+str(testing_sampling)+".csv"
    return filename

def generate_random_charges_insurance(mean=13270.422265141257, std=12110.011236694001):
    return np.random.normal(loc=mean, scale=std, size=1).tolist()[0]

def generate_random_charges_ChanceOfAdmition(mean=0.72435, std=0.14260933017384092):
    return np.random.normal(loc=mean, scale=std, size=1).tolist()[0]

def generate_random_charges_usedcar(mean=50014.51, std=42279.49):
    return np.random.normal(loc=mean, scale=std, size=1).tolist()[0]

def create_IO_example(dataset,x,y,feature_num,Names,config,is_test=False):

    if dataset=="insurance":
        generate_random_charges=generate_random_charges_insurance
    if dataset=="ChanceOfAdmition":
        generate_random_charges=generate_random_charges_ChanceOfAdmition
    if dataset=="usedcars":
        generate_random_charges=generate_random_charges_usedcar
    
    ex_context=""
    for i in range(feature_num):
        feature_name=Names[i]
        if config=="missing_values_0.5" and random.random()>0.5:
            ex_context+=feature_name+": ?\n"
        else:
            ex_context+=feature_name+": "+str(x[i])+"\n"

    output_value=y
    if config=="realname_randomoutput" and not is_test:
        output_value=generate_random_charges()

    return ex_context+Names[-1]+": ", str(output_value)+"\n\n"
