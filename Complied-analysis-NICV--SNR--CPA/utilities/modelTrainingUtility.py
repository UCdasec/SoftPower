import sys

import matplotlib.pyplot as plt
# model training
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# importing loadDataUtility for some functions required for data preprocessing
from loadDataUtility import *


def load_training_data(params):
    """
    This function loads the data required for training the model
    :param params: dictionary containing the parameters required for loading the data
    :return: the traces to be used for training and corresponding labels to those traces
    """
    print('loading data for training model ...')
    target_byte = params['target_byte']
    start_idx, end_idx = params["start_idx"], params["end_idx"]
    fList = os.listdir(params["input_path"])
    print('processing data for key byte', target_byte)

    fileName = ""
    for item in fList:
        if item.startswith('train'):
            fileName = item
    train_data_path = os.path.join(params["input_path"], fileName)
    if '' == train_data_path:
        raise ValueError('file did not found: {}'.format(train_data_path))

    train_data_whole_pack = np.load(train_data_path)
    train_traces, train_label, train_text_in, key = gen_features_labels(train_data_whole_pack, target_byte, start_idx,
                                                                        end_idx)

    print('training data loaded successfully!')
    return train_traces, train_label, key

def load_training_data_2(params):
    """
    This function loads the data required for training the model
    :param params: dictionary containing the parameters required for loading the data
    :return: the traces to be used for training and corresponding labels to those traces
    """
    print('loading data for training model ...')
    target_byte = params['target_byte']
    start_idx, end_idx = params["start_idx"], params["end_idx"]
    file_name = params["input_path"]
    print('processing data for key byte', target_byte)

    try:
        train_data_whole_pack = np.load(file_name)
    except OSError:
        print("could not access {}".format(file_name))
        sys.exit()
        
    train_traces, train_label, train_text_in, key = gen_features_labels(train_data_whole_pack, target_byte, start_idx,
                                                                        end_idx)
    print('training data loaded successfully!')
    return train_traces, train_label, key

def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


# Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, verbose=False):
    # check if directory to save model exists
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    checkpointer = ModelCheckpoint(save_file_name,
                                   monitor='val_accuracy',
                                   verbose=verbose,
                                   # save_weights_only=True,
                                   # save_best_only=True,
                                   mode='max')
    earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    callbacks = [checkpointer, earlyStopper]

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    input_layer_shape = input_layer_shape[0]

    # Sanity check
    if input_layer_shape[1] != X_profiling.shape[1]:
        print("Error: model input shape %d instead of %d is not expected ..." %
              (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a CNN: expand the dimensions
        print('reshaping the data for training CNN ...')
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        print('shape of the training dataset: ', Reshaped_X_profiling.shape)
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)

    Y_profiling = to_categorical(Y_profiling, num_classes=256)
    history = model.fit(x=Reshaped_X_profiling, y=Y_profiling,
                        validation_split=0.1, batch_size=batch_size,
                        verbose=verbose, epochs=epochs,
                        shuffle=True, callbacks=callbacks)

    print('model save to path: {}'.format(save_file_name))
    return history


def load_test_data(params, clsNum=256):
    """
    This function loads the dataset required for testing the model
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
    val_data_path = os.path.join(params["input_path"], fileName)
    if '' == fileName:
        raise ValueError('file did not find: {}'.format(val_data_path))

    val_data_whole_pack = np.load(val_data_path)
    test_traces, test_label, test_text_in, key = gen_features_labels(val_data_whole_pack, target_byte, start_idx,
                                                                     end_idx)

    test_label = to_categorical(test_label, clsNum)
    print('test data loaded successfully!')
    return test_traces, test_label, test_text_in, key

def load_test_data_2(params, clsNum=256):
    """
    This function loads the dataset required for testing the model
    :param params: dictionary containing the parameters of the dataset to be used for testing
    :param clsNum: number of classes in the dataset (here 256)
    :return: Testing power traces along with their labels, plaintext and key
    """
    # checking_tool.check_file_exists(ascad_database_file)
    # in_file = h5py.File(ascad_database_file, "r")
    print('loading the test data ...')
    target_byte = params["target_byte"]
    start_idx, end_idx = params["start_idx"], params["end_idx"]
    file_name = params["input_path"]
    print('processing data for key byte', target_byte)

    try:
        val_data_whole_pack = np.load(file_name)
    except OSError:
        print("could not access {}".format(file_name))
        sys.exit()

    test_traces, test_label, test_text_in, key = gen_features_labels(val_data_whole_pack, target_byte, start_idx,
                                                                     end_idx)

    test_label = to_categorical(test_label, clsNum)
    print('test data loaded successfully!')
    return test_traces, test_label, test_text_in, key


# Compute the rank of the real key for a give set of predictions
def rank(predictions, plaintext_list, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte):
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
                """
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba**2)
                """
                min_proba = 0.0000000000000000000000000000000001
                key_bytes_proba[i] += np.log(min_proba ** 2)

    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank, key_bytes_proba


def full_ranks(model, dataset, key, plaintext_attack, min_trace_idx, max_trace_idx, target_byte, rank_step):
    # Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
    real_key = key[target_byte]
    # Check for overflow
    if max_trace_idx > dataset.shape[0]:
        raise ValueError("Error: asked trace index %d overflows the total traces number %d"
                         % (max_trace_idx, dataset.shape[0]))

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape[0]
    # Sanity check
    if input_layer_shape[1] != dataset.shape[1]:
        raise ValueError("Error: model input shape %d instead of %d is not expected ..." %
                         (input_layer_shape[1], len(dataset[0, :])))

    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        print('# This is a MLP')
        input_data = dataset[min_trace_idx:max_trace_idx, :]
    elif len(input_layer_shape) == 3:
        print('# This is a CNN: reshape the data')
        input_data = dataset[min_trace_idx:max_trace_idx, :]
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        raise ValueError("Error: model input shape length %d is not expected ..." % len(input_layer_shape))

    # Predict our probabilities
    predictions = model.predict(input_data, batch_size=200, verbose=0)

    index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba = rank(predictions[t - rank_step:t],
                                              plaintext_attack[t - rank_step:t],
                                              real_key,
                                              t - rank_step,
                                              t,
                                              key_bytes_proba,
                                              target_byte)
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
    return f_ranks


def get_the_labels(textins, key, target_byte):
    labels = []
    for i in range(textins.shape[0]):
        text_i = textins[i]
        label = aes_internal(text_i[target_byte], key[target_byte])
        labels.append(label)

    labels = np.array(labels)
    return labels


def plot_figure(x, y, model_file_name, dataset_name, fig_save_name, testType, save_fig = True):
    plt.subplots()
    plt.title(model_file_name + ' against ' + dataset_name + ' testType ' + testType)
    plt.xlabel('number of traces')
    plt.ylabel('rank')
    plt.grid(True)
    plt.plot(x, y)
    plt.xlim(xmin=0)
    if save_fig == True:
        plt.savefig(fig_save_name)
    plt.show(block=False)
