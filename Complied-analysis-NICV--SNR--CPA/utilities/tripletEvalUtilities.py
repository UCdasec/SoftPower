import os
import numpy as np

from loadDataUtility import *

from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from joblib import dump, load


def load_data(data_params, data_label = None):
    """
    This function loads the dataset required for training/testing the k-NN
    :param data_params: dictionary containing the parameters of the dataset
    :param data_label: the label of the dataset which is being processed either train or test
    :return: The restructured dataset for training/testing the k-NN
    """

    print("parameters of the dataset are: ", data_params)

    # load the power traces on which base model is to be trained (Pre-training phase)
    if data_label == "train":
        X_profiling, y_profiling, train_plain_text, train_key = load_train_data_nshot(data_params)
        # getting the subset of the dataset
        print('getting subset of the data with %s samples from each class'%data_params['n'])
        x, y, all_data_df = create_df(X_profiling, y_profiling, data_params['n'])
        x = x.reshape((x.shape[0], x.shape[1], 1))
        print("reshaped traces for the feature extractor:", x.shape)
        print("shape of the labels:", y.shape)
    elif data_label == "test":
        # Below commented line is call to the original function. Below that is the call to the updated function.
        # X_profiling, y_profiling, test_plain_text, test_key = load_test_data_nshot(data_params)
        X_profiling, y_profiling, test_plain_text, test_key = load_test_data_nshot_2(data_params)
        x_test_df = pd.DataFrame(X_profiling)
        y_test_df = pd.DataFrame(y_profiling, columns=['label'])
        frames = [y_test_df, x_test_df]
        test_data_df = pd.concat(frames, axis=1)
        x_test = test_data_df.iloc[:, 1:]
        x_test = x_test.to_numpy()
        y_test = test_data_df["label"]
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        print("reshaped traces for the feature extractor:", x_test.shape)
        print("shape of the labels:", y_test.shape)

    # shape of the dataset
    print('shape of X_profiling: ', X_profiling.shape)
    print('shape of y_profiling: ', y_profiling.shape)

    # number of unique classes in the dataset
    nb_classes = len(np.unique(y_profiling))
    print('number of classes in the dataset: ', nb_classes)

    if data_label == "train":
        print('Train key: ', train_key)
        return x, y, all_data_df, nb_classes, train_plain_text, train_key
    elif data_label == "test":
        print('Test key: ', test_key)
        return x_test, y_test, test_data_df, nb_classes, test_plain_text, test_key


def load_train_data_nshot(params):
    """
    This function loads the dataset required for training the model using transfer learning
    :param params: dictionary containing the parameters of the dataset to be used for testing
    :param clsNum: number of classes in the dataset (here 256)
    :return: Testing power traces along with their labels, plaintext and key
    """
    # checking_tool.check_file_exists(ascad_database_file)
    # in_file = h5py.File(ascad_database_file, "r")
    print('loading the training data ...')
    target_byte = params["target_byte"]
    start_idx, end_idx = params["start_idx"], params["end_idx"]

    fileName = ''
    fList = os.listdir(params["input_path"])
    if params["testType"] == 'samekey':
        keyword = 'train_same_key'
    elif params["testType"] == 'diffkey':
        keyword = 'train_diff_key'
    else:
        raise ValueError()

    for item in fList:
        if item.startswith(keyword):
            fileName = item
    train_data_path = os.path.join(params["input_path"], fileName)
    if '' == fileName:
        raise ValueError('file did not find: {}'.format(train_data_path))

    print('train_data_path: ', train_data_path)

    val_data_whole_pack = np.load(train_data_path)
    train_traces, train_label, train_text_in, key = gen_features_labels(val_data_whole_pack, target_byte, start_idx,
                                                                     end_idx)
    print('training data loaded successfully!')
    return train_traces, train_label, train_text_in, key


def load_test_data_nshot(params):
    """
    This function loads the dataset required for testing the model to be trained using transfer learning
    :param params: dictionary containing the parameters of the dataset to be used for testing
    :param clsNum: number of classes in the dataset (here 256)
    :return: Testing power traces along with their labels, plaintext and key
    """
    # checking_tool.check_file_exists(ascad_database_file)
    # in_file = h5py.File(ascad_database_file, "r")
    print('loading the test data ...')
    target_byte = params["target_byte"]
    start_idx, end_idx = params["start_idx"], params["end_idx"]

    fileName = ''
    fList = os.listdir(params["input_path"])
    if params["testType"] == 'samekey':
        keyword = 'test_same_key'
    elif params["testType"] == 'diffkey':
        keyword = 'test_diff_key'
    else:
        raise ValueError()

    for item in fList:
        if item.startswith(keyword):
            fileName = item
    test_data_path = os.path.join(params["input_path"], fileName)
    if '' == fileName:
        raise ValueError('file did not find: {}'.format(test_data_path))
    print('test_data_path: ', test_data_path)
    test_data_whole_pack = np.load(test_data_path)
    test_traces, test_label, test_text_in, key = gen_features_labels(test_data_whole_pack, target_byte, start_idx,
                                                                     end_idx)
    # test_label = to_categorical(test_label, clsNum)
    print('test data loaded successfully!')
    return test_traces, test_label, test_text_in, key

def load_test_data_nshot_2(params):
    """
    This function loads the dataset required for testing the model to be trained using transfer learning
    :param params: dictionary containing the parameters of the dataset to be used for testing
    :param clsNum: number of classes in the dataset (here 256)
    :return: Testing power traces along with their labels, plaintext and key
    """
    # checking_tool.check_file_exists(ascad_database_file)
    # in_file = h5py.File(ascad_database_file, "r")
    print('loading the test data ...')
    target_byte = params["target_byte"]
    start_idx, end_idx = params["start_idx"], params["end_idx"]
    path = params["input_path"]
    
    # conditional logic to check whether the input_path is a file or directory.
    if os.path.isdir(path):
        fileName = ''
        fList = os.listdir(path)
        if params["testType"] == 'samekey':
            keyword = 'test_same_key'
        elif params["testType"] == 'diffkey':
            keyword = 'test_diff_key'
        else:
            raise ValueError()

        for item in fList:
            if item.startswith(keyword):
                fileName = item
                
        test_data_path = os.path.join(params["input_path"], fileName)
        
        if '' == fileName:
            raise ValueError('file did not find: {}'.format(test_data_path))
            
        print('test_data_path: ', test_data_path)
        test_data_whole_pack = np.load(test_data_path)
    elif os.path.isfile(path):
        try:
            test_data_whole_pack = np.load(path)
        except OSError:
            print("could not access {}".format(file_name))
            sys.exit()
        else:
            print('test_data_path: ', path)
    else:
        raise ValueError('did not find: {}'.format(path))
    
    test_traces, test_label, test_text_in, key = gen_features_labels(test_data_whole_pack, target_byte, start_idx,
                                                                     end_idx)
    # test_label = to_categorical(test_label, clsNum)
    print('test data loaded successfully!')
    return test_traces, test_label, test_text_in, key


def load_feature_extractor(feat_extractor_path):
    """
    This function loads the feature extractor to extract the features from the raw traces.
    :param feat_extractor_path: path where the feature extractor is saved
    :return: the feature extractor model
    """
    # loading weights
    print('loading the feature extractor for %s ...'%(feat_extractor_path))
    features_model = load_model(feat_extractor_path)
    print('feature extractor loaded successfully!')
    print(features_model.summary())
    return features_model


def extract_features_train(features_model, x_train):
    """
    This function extracts the features from the feature extractor, which are further used for
    training a k-NN in the attack phase. The features of the training dataset are extracted.
    :param features_model: model to extract features
    :param x_train: training dataset for which features are to be extracted
    :return: (x_train_feature_set)
    """
    # Extracting features for the training dataset
    x_train_feature_set = features_model.predict(x_train)

    return x_train_feature_set


def extract_features_test(features_model, x_test):
    """
    This function extracts the features from the feature extractor, which are further used for
    testing a k-NN in the attack phase. The features of the testing dataset are extracted.
    :param features_model: model to extract features
    :param x_test: testing dataset for which features are to be extracted
    :return: (x_train_feature_set, x_test_feature_set)
    """
    # Extracting features for the training dataset
    x_test_feature_set = features_model.predict(x_test)

    return x_test_feature_set


def train_test_knn(x_train, y_train, x_test, y_test, neighbours=None, eval_type=None):
    """
    This function trains and test the k-NN model.
    :param x_train: the dataset used for training the k-NN
    :param y_train: the labels corresponding to the power traces in x_train
    :param x_test: the dataset used for evaluating the performance of K-NN
    :param y_test: the labels corresponding to the power traces in x_test
    :param neighbours:  Number of neighbors to use by default for kneighbors queries
    :param eval_type: either N-MEV or N-ALL
    :return: prediction of each testing trace
    """
    # ::TODO: Add data manipulation for N-MEV

    knn = KNeighborsClassifier(n_neighbors=neighbours,
                               # n-shot Number of neighbors to use by default for kneighbors queries. n for n-shot learning
                               weights='distance',
                               p=2,  # Power parameter for the Minkowski metric.
                               metric='cosine',  # the distance metric to use for the tree.
                               algorithm='brute'  # Algorithm used to compute the nearest neighbors
                               )
    knn.fit(x_train, y_train)
    accuracy_top_1 = accuracy_score(y_test, knn.predict(x_test))
    print('Accuracy score of k-NN: ', accuracy_top_1)

    test_predictions = []
    for i in range(len(x_test)):
        temp_trace = x_test[i]
        proba = knn.predict_proba([temp_trace])
        test_predictions.append(proba)
        # print('prediction probabilities: ', proba)
    return test_predictions


def train_knn_experiment_e(x_train, y_train, model_path, neighbours=100,  eval_type=None):
    """
    This function trains the k-NN for experiment e and saves it to the model_path.
    :param x_train: Input for training k-NN
    :param y_train: corresponding labels of power traces input to k-NN
    :param model_path: path to save the trained k-NN
    :param neighbours:  Number of neighbors to use by default for kneighbors queries
    :param eval_type: either N-ALL or N-MEV
    :return: None
    """
    # TODO:: Add implementation for N-MEV
    knn = KNeighborsClassifier(n_neighbors=neighbours,
                               # n-shot Number of neighbors to use by default for k-neighbors queries. n for n-shot learning
                               weights='distance',
                               p=2,  # Power parameter for the Minkowski metric.
                               metric='cosine',  # the distance metric to use for the tree.
                               algorithm='brute'  # Algorithm used to compute the nearest neighbors
                               )
    print('Training k-NN model ...')
    # training k-NN model
    knn.fit(x_train, y_train)
    print('k-NN model trained successfully!')
    accuracy_top_1 = accuracy_score(y_train, knn.predict(x_train))
    print('Accuracy score of k-NN on training dataset: ', accuracy_top_1)
    # saving the k-NN model
    dump(knn, model_path)
    print('k-NN model saved to %s.'%(model_path))


def test_knn_experiment_e(x_test, y_test, model_path, eval_type=None):
    """
    This function loads and tests the accuracy of k-NN classifier on the testing dataset.
    :param x_test: input of the k-NN classifier
    :param y_test: corresponding labels for power traces in x_test
    :param model_path: path of the k-NN model
    :param eval_type: type of evaluation, either N-ALL or N-MEV
    :return: the predictions of the k-NN on the testing dataset
    """
    # TODO:: Add implementation for N-MEV

    # load the k-NN model
    knn = load(model_path)

    # getting accuracy on the testing dataset
    accuracy_top_1 = accuracy_score(y_test, knn.predict(x_test))
    print('Accuracy score of k-NN on testing dataset: ', accuracy_top_1)

    # making predictions on test dataset using k-NN
    test_predictions = []
    for i in range(len(x_test)):
        temp_trace = x_test[i]
        proba = knn.predict_proba([temp_trace])
        test_predictions.append(proba)
        # print('prediction probabilities: ', proba)
    return test_predictions


def rank(predictions, plaintext_list, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
    """
    This function computes the rank of the real key for given set of predictions
    :param predictions:
    :param plaintext_list:
    :param real_key:
    :param min_trace_idx:
    :param max_trace_idx:
    :param last_key_bytes_proba:
    :param target_byte:
    :return:
    """
    # Compute the rank
    if len(last_key_bytes_proba) == 0:
        # If this is the first rank we compute, initialize all the estimates to zero
        key_bytes_proba = np.zeros(256)
    else:
        # This is not the first rank we compute: we optimize things by using the
        # previous computations to save time!
        key_bytes_proba = last_key_bytes_proba

    for p in range(0, max_trace_idx - min_trace_idx):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext = plaintext_list[p][target_byte]
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs
            # AES_Sbox[plaintext ^ i]
            tmp_label = aes_internal(plaintext, i)
            proba = predictions[p][tmp_label]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                # We do not want an -inf here, put a very small epsilon
                # that corresponds to a power of our min non zero proba
                '''
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba**2)
                '''
                min_proba = 0.0000000000000000000000000000000001
                key_bytes_proba[i] += np.log(min_proba ** 2)

    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return (real_key_rank, key_bytes_proba)


def generate_ranks(predictions, min_trace_idx, max_trace_idx, plain_text, key, target_byte, step_size):
    """
    This function generates the key rank for the predictions obtained using k-NN
    :param predictions: predictions generated on the testing dataset
    :param key: key used for generating the cipher text
    :param plain_text: plain text used while collecting power traces
    :param min_trace_idx: minimum index of the trace
    :param max_trace_idx: maximum index of the trace
    :param target_byte: target byte
    :param rank_step: step size
    :return: the key ranks for the predictions
    """
    real_key = key[target_byte]

    predictions = np.concatenate(predictions)
    #     print('predictions: ', predictions)

    index = np.arange(min_trace_idx + step_size, max_trace_idx, step_size)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []

    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t - step_size:t], plain_text[t - step_size:t], real_key,
                                              t - step_size, t, key_bytes_proba, target_byte)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks


def generate_plot(x, y, title):
    """
    This figure generates the plot for the key and corresponding ranks generated.
    :param x: the number of traces used
    :param y: rank of the key
    :param title: title of the figure
    :return:
    """
    plt.subplots()
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.title(title)
    plt.plot(x, y)
    plt.xlim(xmin=0)
    plt.ylim(ymin=-0.05)
    # plt.savefig(title, dpi=200)
    plt.show(block=False)