import re
import os
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
import gensim.models.keyedvectors as word2vec
import torch.utils.data as utils
import torch

nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')

dimensions = {
    "IMDB": 52,
    "IMDB_polarised": 28,
    'MR': 28,
    'SST-1': 28,
    'SST-2': 28,
}

# Adapted from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s.strip('\"')
    s.strip('\'')

    # split into words
    tokens = word_tokenize(s)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    return [w for w in words if not w in stop_words]


def create_embedding_index(embedding_selection):
    return {'glove_300': lambda: load_glove('embeddings/glove.6B.300d.txt'),
            'glove_50': lambda: load_glove('embeddings/glove.6B.50d.txt'),
            'word2vec_300': lambda: load_word2vec('embeddings/GoogleNews-vectors-negative300.bin'),
            'fasttext_300': lambda: load_fasttext('embeddings/wiki-news-300d-1M.vec')}[embedding_selection]()


def load_glove(file_name):
    embeddings_index = dict()
    f = open(file_name, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def load_word2vec(file_name):
    embeddings_index = dict()
    word2vecDict = word2vec.KeyedVectors.load_word2vec_format(file_name, binary=True)

    for word in word2vecDict.wv.vocab:
        embeddings_index[word] = word2vecDict.word_vec(word)
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def load_fasttext(file_name):
    embeddings_index = dict()
    f = open(file_name, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def apply_embeddings(x_text, embeddings_index, embedding_dimension, dimension_reduction_method):
    # vec used to convert the collection of text examples to a matrix of token counts
    vec = CountVectorizer(tokenizer=clean_str)

    # Learn the vocabulary dictionary and return the term-document matrix
    vec.fit_transform(x_text)

    # Array mapping from feature integer indices to feature name
    vocab_names = vec.get_feature_names()

    # Pad vector of dimension 50 to 52 if not applying dimensionality reduction
    if embedding_dimension == 50 and dimension_reduction_method == 'None':
        embedding_vector_dim = 52

        # Initialise array to store embedding vector for each word in vocab
        vocab_embeddings_full = np.zeros((len(vocab_names), embedding_vector_dim))
        total_empty = 0

        for i in range(len(vocab_names)):
            # Find the embedding vector (len: embedding dimension) corresponding to the specific word in the corpus
            embedding_vector = embeddings_index.get(vocab_names[i])

            # If word is present in the embedding vector, append the embedding array with this embedding_vector
            if embedding_vector is not None:
                embedding_vector = np.insert(embedding_vector, 0, 0)
                embedding_vector = np.append(embedding_vector, 0)
                vocab_embeddings_full[i] = embedding_vector
            else:
                total_empty += 1

    else:
        embedding_vector_dim = embedding_dimension

        # Initialise array to store embedding vector for each word in vocab
        vocab_embeddings_full = np.zeros((len(vocab_names), embedding_vector_dim))
        total_empty = 0

        for i in range(len(vocab_names)):
            # Find the embedding vector (len: embedding dimension) corresponding to the specific word in the corpus
            embedding_vector = embeddings_index.get(vocab_names[i])

            # If word is present in the embedding vector, append the embedding array with this embedding_vector
            if embedding_vector is not None:
                vocab_embeddings_full[i] = embedding_vector
            else:
                total_empty += 1

    return vocab_names, vocab_embeddings_full, total_empty


def reduce_embedding_dimensions(embedding_selection, vocab_embeddings_full, dimension, output_path):
    return {'None': lambda: reduce_embedding_dimensions_None(vocab_embeddings_full, dimension, output_path),
            'PCA': lambda: reduce_embedding_dimensions_PCA(vocab_embeddings_full, dimension, output_path),
            'IPCA': lambda: reduce_embedding_dimensions_IPCA(vocab_embeddings_full, dimension, output_path),
            'KPCA': lambda: reduce_embedding_dimensions_KPCA(vocab_embeddings_full, dimension, output_path),
            'SVD': lambda: reduce_embedding_dimensions_SVD(vocab_embeddings_full, dimension, output_path),
            'GRP': lambda: reduce_embedding_dimensions_GRP(vocab_embeddings_full, dimension, output_path)}[
        embedding_selection]()


def reduce_embedding_dimensions_None(vocab_embeddings_full, dimension, output_path):
    vocab_embeddings_reduced = vocab_embeddings_full
    np.save(os.path.join(output_path, 'vocab_embeddings'), vocab_embeddings_reduced)

    return vocab_embeddings_reduced


def reduce_embedding_dimensions_PCA(vocab_embeddings_full, dimension, output_path):
    pca = PCA(n_components=dimension)
    vocab_embeddings_reduced = pca.fit_transform(vocab_embeddings_full)
    np.save(os.path.join(output_path, 'vocab_embeddings'), vocab_embeddings_reduced)

    return vocab_embeddings_reduced


def reduce_embedding_dimensions_IPCA(vocab_embeddings_full, dimension, output_path):
    n_batches = 256
    inc_pca = IncrementalPCA(n_components=dimension)
    for batch in np.array_split(vocab_embeddings_full, n_batches):
        inc_pca.partial_fit(batch)
    vocab_embeddings_reduced = inc_pca.transform(vocab_embeddings_full)
    np.save(os.path.join(output_path, 'vocab_embeddings'), vocab_embeddings_reduced)

    return vocab_embeddings_reduced


def reduce_embedding_dimensions_KPCA(vocab_embeddings_full, dimension, output_path):
    kpca = KernelPCA(kernel="rbf", n_components=dimension, gamma=None, fit_inverse_transform=True, random_state=2019,
                     n_jobs=1)
    kpca.fit(vocab_embeddings_full[:10000, :])
    vocab_embeddings_reduced = kpca.transform(vocab_embeddings_full)
    np.save(os.path.join(output_path, 'vocab_embeddings'), vocab_embeddings_reduced)

    return vocab_embeddings_reduced


def reduce_embedding_dimensions_SVD(vocab_embeddings_full, dimension, output_path):
    SVD = TruncatedSVD(n_components=dimension, algorithm='randomized', random_state=2019, n_iter=5)
    SVD.fit(vocab_embeddings_full[:10000, :])
    vocab_embeddings_reduced = SVD.transform(vocab_embeddings_full)
    np.save(os.path.join(output_path, 'vocab_embeddings'), vocab_embeddings_reduced)

    return vocab_embeddings_reduced


def reduce_embedding_dimensions_GRP(vocab_embeddings_full, dimension, output_path):
    GRP = GaussianRandomProjection(n_components=dimension, eps=0.5, random_state=2019)
    GRP.fit(vocab_embeddings_full[:10000, :])
    vocab_embeddings_reduced = GRP.transform(vocab_embeddings_full)
    np.save(os.path.join(output_path, 'vocab_embeddings'), vocab_embeddings_reduced)

    return vocab_embeddings_reduced


def load_pretrained_embeddings(embedding_dimension_path):
    vocab_embeddings_reduced = np.load(os.path.join(embedding_dimension_path, 'vocab_embeddings.npy'))
    return vocab_embeddings_reduced


def select_embedded_features(x_text, dataset, embedding_type, embedding_reduction, vocab_names, vocab_embeddings_reduced):
    features_path = os.path.join(os.path.join(os.path.join(os.getcwd(), 'datasets_prepared'), dataset), embedding_type)
    if os.path.exists(features_path):
        x_text_prepared = np.load(os.path.join(features_path, 'x_text_prepared.npy'))
    else:
        os.makedirs(features_path)

    # Remove stop words and punctuation etc. from sentences according to clean_str method
    tokenised_sentences = [clean_str(sentence) for sentence in x_text]

    # Initialise output vector (4-dimensional)
    x_text_prepared = np.zeros((len(tokenised_sentences), 1, embedding_reduction, embedding_reduction))

    for sentence_idx, sentence in enumerate(tokenised_sentences):
        text = np.zeros((embedding_reduction, embedding_reduction))

        # Iterate over the words in the tokenised sentences up to the maximum dimension
        for word_idx in range(min(embedding_reduction, len(sentence))):
            # Check if the word is present in the embeddings dictionary
            if sentence[word_idx] in vocab_names:
                # Use the word to find the index in vocab_names and use this to find the vector
                text[word_idx] = vocab_embeddings_reduced[vocab_names.index(sentence[word_idx])]

        # Add the found vector to the output vector at the position of each sentence in turn
        x_text_prepared[sentence_idx][0] = text

        np.save(os.path.join(features_path, 'x_text_prepared'), x_text_prepared)

    return x_text_prepared


def create_batches(x_text_prepared_all, y_labels_all, batch_size, training_set_percentage):
    torch.cuda.empty_cache()

    parameters = {'batch_size': batch_size,
              'drop_last': True,
              'shuffle': True,
              'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    # Make validation sets
    x_train, x_test, y_train, y_test = train_test_split(x_text_prepared_all, y_labels_all, test_size=training_set_percentage, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15,
                                                      random_state=42)

    # Transform features into torch tensors
    x_train_tensor = torch.stack([torch.Tensor(i) for i in x_train])
    x_val_tensor = torch.stack([torch.Tensor(i) for i in x_val])
    x_test_tensor = torch.stack([torch.Tensor(i) for i in x_test])

    # Transform labels into torch tensors
    y_train_tensor = torch.from_numpy(y_train)
    y_val_tensor = torch.from_numpy(y_val)
    y_test_tensor = torch.from_numpy(y_test)

    # Create test, train and val datasets
    train_dataset = utils.TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = utils.TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = utils.TensorDataset(x_test_tensor, y_test_tensor)

    # Create batches
    train_dataloader = utils.DataLoader(train_dataset, **parameters)
    val_dataloader = utils.DataLoader(val_dataset, **parameters)
    test_dataloader = utils.DataLoader(test_dataset, **parameters)

    return train_dataloader, val_dataloader, test_dataloader


def create_batches_model_apply(x_text_prepared, y_labels, batch_size):
    torch.cuda.empty_cache()

    parameters = {'batch_size': batch_size,
              'drop_last': True,
              'shuffle': True,
              'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    # Transform features into torch tensors
    inspect_x = torch.stack([torch.Tensor(i) for i in x_text_prepared])

    # Transform labels into torch tensors
    inspect_y = torch.from_numpy(y_labels)

    # Create test, train and dev datasets
    inspect_data = utils.TensorDataset(inspect_x, inspect_y)

    # Create batches
    inspect_dataloader = utils.DataLoader(inspect_data, **parameters)

    return inspect_dataloader