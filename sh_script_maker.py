import os

# All options
# dataset_list = ['IMDB']
# embedding_list = ["fasttext_300", "glove_300", "glove_50", "word2vec_300"]
# dim_method_list = ["PCA", "SVD", "None"]
# optim_list = ["adabound", "adam", "adagrad"]
# training_percentage_list = ["0.999", "0.8", "0.5", "0.2", "0.0001"]
# embedding_reduction_list = ["28", "52"]

dataset_list = ['IMDB']
embedding_list = ["glove_300"]
dim_method_list = ["PCA"]
optim_list = ["adam", "adabound"]
training_percentage_list = ["0.999", "0.0001"]
embedding_reduction_list = ["52"]


def create_batch_files(dataset_list, embedding_list, dim_method_list, optim_list, training_percentage_list,
                       embedding_reduction_list):
    script_list = ""
    script_save_location = "./"
    if not os.path.exists(script_save_location):
        os.makedirs(script_save_location)

    for dim_method in dim_method_list:
        if dim_method == "none":
            embedding_list = ["glove_50"]
            embedding_reduction_list = ["28"]
        for dataset in dataset_list:
            for embedding in embedding_list:
                for optim in optim_list:
                    for percentage in training_percentage_list:
                        for embedding_reduction in embedding_reduction_list:
                            text = ("#! /bin/bash \n" +
                                    "#SBATCH --job-name='{}_{}_{}_{}_{}_{}'".format(
                                        dataset, embedding, dim_method, optim, percentage, embedding_reduction)
                                    + "\n" +
                                    "#SBATCH --mail-type=ALL \n" +
                                    "#SBATCH --mail-user=stephanie.jury@city.ac.uk \n" +
                                    "#SBATCH --nodes=1 \n" +
                                    "#SBATCH --ntasks-per-node=8 \n" +
                                    "#SBATCH --output job%J.out \n" +
                                    "#SBATCH --error job%J.err \n" +
                                    "#SBATCH --partition=normal \n" +
                                    "#SBATCH --gres=gpu:2 \n" +
                                    "module load cuda/10.0 \n" +
                                    "python3/intel \n" +

                                    "python3 'CapsNet_train_test.py' '{}' '{}' '{}' '{}' '{}' '{}' > {}_{}_{}_{}_{}_{}.txt".format(
                                        dataset, embedding, dim_method, optim, percentage, embedding_reduction,
                                        dataset, embedding, dim_method, optim, percentage, embedding_reduction,
                                        dataset, embedding, dim_method, optim, percentage, embedding_reduction))

                            script_list += "sbatch {}_{}_{}_{}_{}_{}.sh \n".format(
                                dataset, embedding, dim_method, optim, percentage, embedding_reduction)
                            f = open("{}/{}_{}_{}_{}_{}_{}.sh".format(
                                script_save_location, dataset, embedding, dim_method, optim, percentage,
                                embedding_reduction), "w+")
                            f.write(text)
                            f.close()

    f = open(os.path.join(script_save_location, "sh_script_list.txt"), "w+")
    f.write(script_list)
    f.close()


create_batch_files(dataset_list, embedding_list, dim_method_list, optim_list, training_percentage_list,
                   embedding_reduction_list)
