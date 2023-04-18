"""
This script contains utility functions required for training and evaluating 256 k-NN classifiers.
"""

from loadDataUtility import *
from modelTrainingUtility import *
from tripletEvalUtilities import *

def gen_features_labels_knn_256(data, input_key_byte, start_index, end_index):
    """
    This function generates the labels and features pair for training a model.
    The labels are generated based on XOR operation between the plaintext byte and the key byte.
    """
    # for loading my dataset
    power_traces, plain_text, key = data['power_trace'], data['plain_text'], data['key']

    # key byte is value between 0 to 15
    labels = []
    for i in range(plain_text.shape[0]):
        text_i = plain_text[i]
        label = aes_internal(text_i[input_key_byte],  key[i][input_key_byte])
        labels.append(label)

    labels = np.array(labels)
    if not isinstance(power_traces, np.ndarray):
        power_traces = np.array(power_traces)
    power_traces = power_traces[:, start_index:end_index]
    return power_traces, labels, plain_text, key


def load_data_knn_256(data_params, data_label=None):
    """
    This function loads and prepared the dataset required for training 256 different classifiers. Each for different key byte value.
    The target key byte for this experiment is only the first key byte.
    """
    print('loading data for training k-NN model ...')
    target_byte = data_params['target_byte']
    start_idx, end_idx = data_params["start_idx"], data_params["end_idx"]
    fList = os.listdir(data_params["input_path"])
    print('processing data for key byte', target_byte)

    if data_label == 'train':
        fileName = ""
        for item in fList:
            if item.startswith('train'):
                fileName = item
        train_data_path = os.path.join(data_params["input_path"], fileName)
        print('training dataset loaded form: ', fileName)
        if '' == train_data_path:
            raise ValueError('file not found: {}'.format(train_data_path))

        train_data_whole_pack = np.load(train_data_path)
        train_traces, train_label, train_text_in, key = gen_features_labels_knn_256(train_data_whole_pack,
                                                                                    target_byte,
                                                                                    start_idx, end_idx)

        x, y, all_data = create_df(train_traces, train_label, data_params['n'])
        x = x.reshape(x.shape[0], x.shape[1], 1)
        print('reshaped traces for feature extractor ', x.shape)
        print('shape of the labels: ', y.shape)
        return x, y, train_text_in

    if data_label == 'test':
        fileName = ""
        for item in fList:
            if item.startswith('test'):
                fileName = item
        test_data_path = os.path.join(data_params["input_path"], fileName)
        print('testing dataset loaded form: ', fileName)
        if '' == test_data_path:
            raise ValueError('file not found: {}'.format(test_data_path))

        test_data_whole_pack = np.load(test_data_path)
        test_traces, test_label, test_text_in, key = gen_features_labels_knn_256(test_data_whole_pack,
                                                                                 target_byte,
                                                                                 start_idx, end_idx)

        x_test_df = pd.DataFrame(test_traces)
        y_test_df = pd.DataFrame(test_label, columns=['label'])
        frames = [y_test_df, x_test_df]
        test_data_df = pd.concat(frames, axis=1)
        x_test = test_data_df.iloc[:, 1:]
        x_test = x_test.to_numpy()
        y_test = test_data_df["label"]
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        print("reshaped traces for the feature extractor:", x_test.shape)
        print("shape of the labels:", y_test.shape)

        return x_test, y_test, test_text_in, key


def train_knn_256(x_train, y_train, class_label, model_path=None, neighbours=None, eval_type=None):
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

    model_name = 'knn-class-' + str(class_label) + '.model'
    model_path = os.path.join(model_path, model_name)
    dump(knn, model_path)

    if (class_label % 25) == 0:
        print('model for class %d saved successfully!' % (class_label))

def test_knn_256(x_test, y_test, model_path, eval_type=None):
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


def generate_plot_knn_256(x, y, title, figure_path, save_figure=False):
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
    if save_figure:
        plt.savefig(figure_path, dpi=96)
        plt.close()
#     plt.show(block=False)