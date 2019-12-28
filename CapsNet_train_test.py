from CapsNet_architecture import *
from CapsNet_model_utils import *
from data_loader import load_data, dataset_num_classes
from text_preparation import *
from environment_utils import *


def model_train(dataset, embedding_type, dimension_reduction_method, optimiser_selection, training_set_percentage,
                embedding_reduction):
    # Set output path for saving trained model, embeddings and performance data
    dataset_level, embedding_level, dimension_reduction_method_level, optimiser_level, training_set_percentage_level, embedding_reduction_level = set_output_path(
        dataset, embedding_type, dimension_reduction_method, optimiser_selection, training_set_percentage,
        embedding_reduction)

    embedding_dimension = int(embedding_type.split("_")[1])

    # Set device (CPU/GPU) and seed for reproducability
    device = set_environment_variables(seed)

    # Load prepared data. If not present convert into torch tensors, save and load.
    x_text, y_labels = load_data(dataset)

    # Create embedding matrix according to selection
    embeddings_index = create_embedding_index(embedding_type)

    # Encode text data according to embedding matrix
    vocab_names, vocab_embeddings_full, total_empty = apply_embeddings(x_text, embeddings_index,
                                                                       embedding_dimension,
                                                                       dimension_reduction_method)

    # Reduce dimensionality of emdedded data to fit on GPUs. Must be less than 60.
    vocab_embeddings_reduced = reduce_embedding_dimensions(dimension_reduction_method, vocab_embeddings_full,
                                                           int(embedding_reduction),
                                                           optimiser_level)

    # Make data into shape of image and unsqueeze: [len(tokenised_sentences), 1, reduced data, reduced data)]
    x_text_prepared = select_embedded_features(x_text, dataset, embedding_type, int(embedding_reduction), vocab_names,
                                               vocab_embeddings_reduced)

    # Create batches of prepared data for model training
    train_dataloader, val_dataloader, test_dataloader = create_batches(x_text_prepared, y_labels, batch_size,
                                                                       (1 - float(training_set_percentage)))

    # Initialise model and save parameters
    # Set model parameters
    criterion = SpreadLoss(num_class=dataset_num_classes[dataset], m_min=0.2, m_max=0.9)
    scheduler_selection = 'ReduceLROnPlateau'
    model = capsules(A=A, B=B, C=C, D=D, E=dataset_num_classes[dataset], iters=2).to(device)
    save_model_info(model, optimiser_level)

    # Train and test model
    model_train_and_test(checkpoint_frequency, epochs, train_dataloader, val_dataloader, test_dataloader, model,
                         scheduler_selection,
                         criterion,
                         optimiser_selection,
                         device, optimiser_level)

# Set environment parameters
environment = 'Local'

# Set training parameters
seed = 42
batch_size = 16 #16 for 52 IMDb #64 for all others
epochs = 20
checkpoint_frequency = 1

# Set model parameters
A, B, C, D = 64, 8, 16, 16  # 97.1
# A, B, C, D = 32, 32, 32, 32 #99.3

if environment == 'Server':
    # Get batch script input parameters
    dataset, embedding_type, dimension_reduction_method, optimiser_selection, training_set_percentage, embedding_reduction = get_user_inputs()
    model_train(dataset, embedding_type, dimension_reduction_method, optimiser_selection, training_set_percentage,
                embedding_reduction)

else:
    dataset_list = ['IMDB', 'SST-1', 'SST-2', 'MR', 'IMDB_polarised']
    embedding_type_list = ['glove_300', 'fasttext_300', 'word2vec_300', 'glove_50']
    dimension_reduction_method_list = ['PCA', 'IPCA', 'KPCA', 'SVD', 'GRP', 'None']
    optimiser_selection_list = ['adam', 'adabound', 'adagrad', 'SGD']
    training_set_percentage_list = [0.8]  # Set percentage of data to train on
    embedding_reduction_list = [28, 52]

    for dataset in dataset_list:
        for embedding_type in embedding_type_list:
            for dimension_reduction_method in dimension_reduction_method_list:
                for optimiser_selection in optimiser_selection_list:
                    for training_set_percentage in training_set_percentage_list:
                        for embedding_reduction in embedding_reduction_list:
                            if __name__ == '__main__':
                                model_train(dataset, embedding_type, dimension_reduction_method, optimiser_selection,
                                            training_set_percentage, embedding_reduction)
