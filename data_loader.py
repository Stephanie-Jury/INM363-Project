# These functions transform the raw datasets into consistent arrays of text and labels
import numpy as np
import os
import re

dataset_num_classes = {
    "IMDB": 2,
    "IMDB_polarised": 2,
    'MR': 2,
    'SST-1': 5,
    'SST-2': 2,
}

def load_data(dataset_selection, input='./datasets_raw'):
    prepared_data_path = os.path.join(os.path.join(os.getcwd(), 'datasets_array'), dataset_selection)

    if os.path.exists(prepared_data_path):
        x_text = np.load(prepared_data_path + '/x_text.npy')
        y_labels = np.load(prepared_data_path + '/y_labels.npy')
        return x_text, y_labels

    else:
        os.makedirs(prepared_data_path)
        return \
            {'MR': lambda: prepare_save_load_mr(dataset_selection, input + '/MR/MR/rt-polarity.pos',
                                   input + '/MR/MR/rt-polarity.neg'),
             'IMDB': lambda: prepare_save_load_imdb(dataset_selection, input + '/IMDB/IMDB'),
             'SST-1': lambda: prepare_save_load_sst1(dataset_selection, input + '/SST-1/train.csv', input + '/SST-1/dev.csv',
                                        input + '/SST-1/test.csv'),
             'SST-2': lambda: prepare_save_load_sst2(dataset_selection, input + '/SST-2/train.csv', input + '/SST-2/dev.csv',
                                        input + '/SST-2/test.csv'),
             'IMDB_polarised': lambda: prepare_save_load_imdb_polarised(dataset_selection,
                                                           input + '/IMDB_polarised/IMDB_polarised/imdb_labelled.txt')}[
                dataset_selection]()


def save_and_load_data(dataset, x_text, y_labels):
    save_location = os.path.join('./datasets_array', dataset)
    np.save(save_location + '/x_text', x_text)
    np.save(save_location + '/y_labels', y_labels)
    x_text, y_labels = load_data(dataset, input='./datasets_raw')
    return x_text, y_labels

# Adapted from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def prepare_save_load_imdb(dataset_selection, folder):
    x_text = list()
    y_labels = list()

    for file in os.listdir(folder + '/test/pos'):
        review_file = open(folder + '/test/pos/' + file, 'r', encoding='utf-8')
        x_text.append(review_file.readline())
        y_labels.append(1)
        review_file.close()

    for file in os.listdir(folder + '/test/neg'):
        review_file = open(folder + '/test/neg/' + file, 'r', encoding='utf-8')
        x_text.append(review_file.readline())
        y_labels.append(0)
        review_file.close()

    for file in os.listdir(folder + '/train/pos'):
        review_file = open(folder + '/train/pos/' + file, 'r', encoding='utf-8')
        x_text.append(review_file.readline())
        y_labels.append(1)
        review_file.close()

    for file in os.listdir(folder + '/train/neg'):
        review_file = open(folder + '/train/neg/' + file, 'r', encoding='utf-8')
        x_text.append(review_file.readline())
        y_labels.append(0)
        review_file.close()

    # Perform manual shuffle
    shuffle_idx = np.random.permutation(len(x_text))
    x_text = [x_text[i] for i in shuffle_idx]
    y_labels = [y_labels[i] for i in shuffle_idx]
    y_labels = np.array(y_labels)

    x_text, y_labels = save_and_load_data(dataset_selection, x_text, y_labels)

    return x_text, y_labels

# Adapted from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def prepare_save_load_mr(dataset_selection, pos, neg):
    positive_examples_raw = list(open(pos, encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples_raw]
    negative_examples_raw = list(open(neg, encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples_raw]

    # Split by words
    x_text = negative_examples + positive_examples

    # Generate labels
    negative_labels = [0 for _ in negative_examples]
    positive_labels = [1 for _ in positive_examples]
    y_labels = np.concatenate([negative_labels, positive_labels], 0)

    x_text, y_labels = save_and_load_data(dataset_selection, x_text, y_labels)

    return x_text, y_labels

# Adapted from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def prepare_save_load_sst1(dataset_selection, train, dev, test):
    x_text = list()
    y_labels = list()

    # Split by words
    for line in [line.split(',', 1) for line in open(train, encoding='utf-8').readlines()]:
        y_labels.append(int(line[0]) - 1)
        x_text.append(line[1])

    for line in [line.split(',', 1) for line in open(dev, encoding='utf-8').readlines()]:
        y_labels.append(int(line[0]) - 1)
        x_text.append(line[1])

    for line in [line.split(',', 1) for line in open(test, encoding='utf-8').readlines()]:
        y_labels.append(int(line[0]) - 1)
        x_text.append(line[1])

    y_labels = np.array(y_labels)

    x_text, y_labels = save_and_load_data(dataset_selection, x_text, y_labels)

    return x_text, y_labels

# Adapted from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def prepare_save_load_sst2(dataset_selection, train, dev, test):
    x_text = list()
    y_labels = list()

    # Split by words
    for line in [line.split(',', 1) for line in open(train, encoding='utf-8').readlines()]:
        y_labels.append(int(line[0]) - 1)
        x_text.append(line[1])

    for line in [line.split(',', 1) for line in open(dev, encoding='utf-8').readlines()]:
        y_labels.append(int(line[0]) - 1)
        x_text.append(line[1])

    for line in [line.split(',', 1) for line in open(test, encoding='utf-8').readlines()]:
        y_labels.append(int(line[0]) - 1)
        x_text.append(line[1])

    y_labels = np.array(y_labels)
    x_text, y_labels = save_and_load_data(dataset_selection, x_text, y_labels)

    return x_text, y_labels

# Adapted from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def prepare_save_load_imdb_polarised(dataset_selection, data):
    x_text = list()
    y_labels = list()

    for line in open(data, encoding='utf-8').readlines():
        split = re.split(r'\s{3,}', line)
        x_text.append(split[0])
        y_labels.append(int(split[1]))

    y_labels = np.array(y_labels)
    x_text, y_labels = save_and_load_data(dataset_selection, x_text, y_labels)

    return x_text, y_labels


def generate_imbalance(x_text, y_labels, majority_class_ratio):
    while len(y_labels[y_labels==0])> len(y_labels) * majority_class_ratio:
        if y_labels[0] == 0:
            y_labels = np.delete(y_labels, 0)
            x_text = np.delete(x_text, 0)
    return