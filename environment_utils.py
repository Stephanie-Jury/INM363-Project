import torch
import random
import numpy as np
import torch.nn as nn
import os
import sys

def get_user_inputs():
    try:
        dataset = sys.argv[1]
        embedding_type = sys.argv[2]
        dimension_reduction_method = sys.argv[3]
        optimiser_selection = sys.argv[4]
        training_set_percentage = sys.argv[5]
        embedding_reduction = sys.argv[6]
        print(
            "Using supplied arguments\n"
            "Dataset: {}\n"
            "Embedding type: {}\n"
            "Dimension reduction method: {}\n"
            "Optimiser: {}\n"
            "Data size: {}\n"
            "Embedding reduction size: {}".format(
                dataset, embedding_type,
                dimension_reduction_method,
                optimiser_selection, training_set_percentage,
                embedding_reduction))

    except:
        dataset = 'MR'
        embedding_type = 'glove_50'
        dimension_reduction_method = 'None'
        optimiser_selection = 'adam'
        training_set_percentage = 0.001  # Set percentage of data to train and validate on (20% held out for testing)
        embedding_reduction = 52
        print(
            "Error reading arguments, using defaults \n"
            "Dataset: {}\n"
            "Embedding type: {}\n"
            "Dimension reduction method: {}\n"
            "Optimiser: {}\n"
            "Data size: {}\n"
            "Embedding reduction size: {}".format(
                dataset, embedding_type,
                dimension_reduction_method,
                optimiser_selection, training_set_percentage,
                embedding_reduction))

    return dataset, embedding_type, dimension_reduction_method, optimiser_selection, training_set_percentage, embedding_reduction

def set_environment_variables(seed):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return device


def set_output_path(dataset, embedding_type, dimension_reduction_method, optimiser, training_set_percentage,
                    embedding_reduction):
    output_root = os.path.join(os.getcwd(), 'model_output')
    dataset_level = os.path.join(output_root, dataset)
    training_set_percentage_level = os.path.join(dataset_level, str(training_set_percentage))
    dataset_dim_level = os.path.join(training_set_percentage_level, str(embedding_reduction))
    embedding_level = os.path.join(dataset_dim_level, embedding_type)
    dimension_reduction_method_level = os.path.join(embedding_level, dimension_reduction_method)
    optimiser_level = os.path.join(dimension_reduction_method_level, optimiser)

    if not os.path.exists(optimiser_level):
        os.makedirs(optimiser_level)

    return dataset_level, embedding_level, dimension_reduction_method_level, optimiser_level, training_set_percentage_level, dataset_dim_level


def select_trained_model_path(dataset, embedding_type, dimension_reduction_method, optimiser):
    output_root = os.path.join(os.getcwd(), 'model_output')
    dataset_level = os.path.join(output_root, dataset)
    embedding_level = os.path.join(dataset_level, embedding_type)
    dimensionality_level = os.path.join(embedding_level, dimension_reduction_method)
    optimiser_level = os.path.join(dimensionality_level, optimiser)
    model_state_dict = nn.load_state_dict(os.path.join(optimiser_level, 'model_100.pth'))
    return optimiser_level, model_state_dict

# Save the model information to file
def save_model_info(model, output_path):
    model_file = open(output_path + "/model.txt", "w")
    model_file.write('Model:\n{}\n'.format(model))
    model_file.write('Total number of parameters:{}\n'.format(sum(p.numel() for p in model.parameters())))
    model_file.write('Total number of trainable parameters:{}\n'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model_file.close()
