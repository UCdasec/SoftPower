"""
This script generates graphs for the 256 k-NN approach.
"""
import matplotlib.pyplot as plt
import numpy as np

from tripletKnnUtilities import *

from sklearn.utils import shuffle
plt.rcParams.update({'font.size': 14})

def select_total_traces(n, x_test, y_test, test_plain_text):
    """
    This function obtains n traces, their corresponding labels, and plaintext for testing purposes
    :param n: number of traces to be used for testing
    :param x_test: power measurements
    :param y_test: labels corresponding to the power measurements
    :param test_plain_text: plain text corresponding to the power measurements.
    :return: n samples of x_test, y_test, and plain text respectively.
    """
    x_test = x_test[:n, :]
    y_test = y_test[:n]
    test_plain_text = test_plain_text[:n, :]

    print('shape of x_test: ', x_test.shape)
    print('shape of y_test: ', y_test.shape)
    print('shape of plaintext: ', test_plain_text.shape)

    return x_test, y_test, test_plain_text


def read_ranks(ranks_path):
    """
    This function is used to read the ranks
    :param ranks_path:
    :return: the ranks for the particular file
    """


def prepare_data_and_get_ranks_knn(data_params):
    """
    This function prepared the dataset and generates the ranking curve
    :param data_params: parameters of the dataset.
    :return: path at which ranking file is saved.
    """
    # load testing dataset
    x_test, y_test, test_data_df, nb_classes_test, test_plain_text, test_key = load_data(data_params, data_label="test")
    print('-' * 90)

    # getting n traces for testing
    print('getting %d traces for testing.'%(data_params["n"]))
    x_test, y_test, plain_text_test = select_total_traces(data_params["n"], x_test, y_test, test_plain_text)

    # loading the feature extractor for extracting features of the power traces for the XMEGA
    print('loading the feature extractor ...')
    input_shape = (data_params['end_idx'] - data_params['start_idx'], 1)
    emb_size = 512
    feat_extractor_path = data_params["feat_extractor_path"]
    feature_extractor_model_xmega = load_feature_extractor(feat_extractor_path)
    print('feature extractor loaded successfully!')
    print('-' * 90)

    # extracting features from testing set
    print('extracting feature for the testing data (x_test) ...')
    x_test_feature_set = extract_features_test(feature_extractor_model_xmega, x_test)
    print('features extracted successfully!')
    print('shape of x_test_feature set: ', x_test_feature_set.shape)
    print('-'*90)

    # trained knn models path
    models_path = data_params['knn_models_path']
    print('k-nn models path: ', models_path)
    ranking_curve_path = data_params["ranking_curve_path"]
    print('key ranking curves will be saved at %s'%(ranking_curve_path))
    if not os.path.isdir(ranking_curve_path):
        os.makedirs(ranking_curve_path)

    eval_type = "N-ALL"  # or N-ALL or N-MEV
    print('making predictions with the k-nn models ...')
    stats = []  # save model name and accuracy

    print('generating ranks and accuracies ...')
    for i in range(0, 256):
        print('#' * 90)

        temp_model_name = 'knn-key-value-' + str(i) + '.model'
        print('making predictions with model: ', temp_model_name)
        temp_model = os.path.join(models_path, temp_model_name)

        ranks_temp = {}
        for j in range(0, 5):
            print('generating ranks for %d iteration ...'%(j))
            min_trace_idx = 0
            max_trace_idx = len(x_test_feature_set)
            step_size = 1

            # generate predictions on the test dataset

            temp_x_test_feature_set, temp_y_test, temp_test_plain_text = shuffle(x_test_feature_set, y_test, plain_text_test)
            test_predictions, accuracy = test_knn_256_graph(temp_x_test_feature_set,
                                                            temp_y_test,
                                                            temp_model,
                                                            eval_type=eval_type)

            temp_test_predictions = test_predictions

            ranks = generate_ranks(temp_test_predictions,
                                   min_trace_idx,
                                   max_trace_idx,
                                   temp_test_plain_text,
                                   test_key,
                                   data_params["target_byte"],
                                   step_size)


            x_temp = {}
            x = [ranks[k][0] for k in range(0, ranks.shape[0])]
            x_temp['x'+str(i)+'-'+data_params["target_board"]] = x
            y_temp = [ranks[k][1] for k in range(0, ranks.shape[0])]
            ranks_temp['y'+str(j)+'-'+data_params["target_board"]] = y_temp

        # save the results
        stats.append([temp_model_name, accuracy])

        if data_params["testType"] == "samekey":
            f_name = data_params["target_board"] + "-same-key-" + str(i) + "-knn-model"
        else:
            f_name = data_params["target_board"] + "-diff-key-" + str(i) + "-knn-model"

        print('saving ranks to csv file ...')
        ranks_dir = os.path.join(ranking_curve_path, 'ranks')
        ranks_file_path = os.path.join(ranking_curve_path + 'ranks', f_name + '-ranks.csv')

        if not os.path.isdir(ranks_dir):
            os.makedirs(ranks_dir)

        ranks_x_df = pd.DataFrame.from_dict(x_temp)
        ranks_y_df = pd.DataFrame.from_dict(ranks_temp)
        ranks_y_avg_df = ranks_y_df.iloc[:, :].mean(axis=1)
        ranks_y_df["y-average"] = ranks_y_avg_df
        ranks_df = pd.concat([ranks_x_df, ranks_y_df], axis=1)
        print('shape of the xmega_df', ranks_df.shape)
        ranks_df.to_csv(ranks_file_path, index=False)
        print('ranks saved to csv file successfully at %s!'%(ranks_file_path))

        print('#'*90)

    csv_file_path = os.path.join(ranking_curve_path, '256-knn-accuracies.csv')
    stats_df = pd.DataFrame(stats, columns=['model-name', 'accuracy'])
    stats_df.to_csv(csv_file_path, index=False)
    print('accuracies for all the 256 knn saved to csv file successfully at %s!' % (csv_file_path))

    return None


def test_knn_256_graph(x_test, y_test, model_path, eval_type=None):
    """
    This function loads and tests the accuracy of k-NN classifier on the testing dataset.
    :param x_test: input of the k-NN classifier
    :param y_test: corresponding labels for power traces in x_test
    :param model_path: path of the k-NN model
    :param eval_type: type of evaluation, either N-ALL or N-MEV
    :return: the predictions of the k-NN on the testing dataset
    """

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
    return test_predictions, accuracy_top_1


def legend_without_duplicate_labels(ax, loc=None):
    """
     This is utility function for removing duplicates from the legend of the plot.
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc, ncol=1)
    # ax.legend(*zip(*unique), bbox_to_anchor=(0,1.02,1,0.2), loc="lower right",
    #             mode="expand", borderaxespad=0, ncol=2)

def generate_ranking_plot(params, key, accuracies_df, index_range, save_path=None):
    """
    This function plots the key ranking for all the 256 k-NNs
    :param ranks_results_path:  path to the key ranking csv files for the 256 k-NNs
    :return: None
    """

    print('generating a plot for %s target board...'%(params['target_board']))
    fig, ax = plt.subplots()

    if params["n"] == 100:
        marker_every = 20
    elif params["n"] == 10000:
        marker_every = 500
    else:
        marker_every = 50

    for i in range(index_range[0], index_range[1]):

        # temp_ranks_file = xmega_unmasked_acc['model-name']
        # print('processing index: ', i)
        temp_ranks_path = os.path.join(params["ranking_curve_path"], 'ranks')
        temp_ranks_path = os.path.join(temp_ranks_path, params['target_board'] + '-same-key-' + str(i) + '-knn-model-ranks.csv')

        temp_ranks = pd.read_csv(temp_ranks_path)
        temp_ranks = temp_ranks.iloc[:params['n'], :]
        temp_x = temp_ranks['x'+str(i)+'-' + params['target_board']]
        temp_y = temp_ranks['y-average']

        if key[params["target_byte"]] >= index_range[0] and key[params["target_byte"]] <= index_range[1]:
            if i == key[params["target_byte"]]:
                ax.plot(temp_x, temp_y, color='red', marker='x', markersize=14, markevery=marker_every, alpha=1.0, label='C_' + str(key[params["target_byte"]]))
            elif i != key[params["target_byte"]]:
                temp_label = 'C_' + str(index_range[0]) + ' to C_' + str(index_range[1]-1) + ' (except C_' + str(
                    key[params["target_byte"]]) + ')'
                ax.plot(temp_x, temp_y, color='grey', linestyle='-', linewidth=1, alpha=0.9, label=temp_label)
        else:
            temp_label = 'C_' + str(index_range[0]) + ' to C_' + str(index_range[1]-1)
            ax.plot(temp_x, temp_y, color='grey', linestyle='-', linewidth=1, alpha=0.9, label=temp_label)


    plt.xlabel('Number of traces')
    plt.ylabel('Mean rank')
    # plt.title(title)
    # plt.grid(True)
    plt.xlim(-1.0, params["n"])
    # ax.set_ylim(-10.0, 300)
    plt.yticks([0, 64, 128, 192, 256])
    legend_without_duplicate_labels(ax, loc="upper center")
    # plt.axis('tight')

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print('plot for %s target board saved successfully.' % (params['target_board']))

    # plt.show()
    plt.close()

def read_csv_file_as_df(csv_file):
    '''
    This functions accepts a csv_file as input and returns the corresponding Pandas DataFrame object.
    '''
    try:
        df_obj = pd.read_csv(csv_file)
    except OSError:
        print("could not access {}".format(csv_file))
        sys.exit()
    
    df_obj.head()
    return df_obj

def get_file_name_from_path(file_path):
    '''
    This functions gets the file name from the input path parameter
    '''
    # The file path is partitioned (beginning from the right of the string) based on the "/" character.
    # The last element of the resulting list represents the .csv file name.
    # This element is then split based on the "." character.
    # This removes the ".csv" extention and returns the file's name.
    file_name = file_path.rpartition('/')[-1]
    file_name = file_name.split(".")[0]
    return file_name

def replace_file_name_text(fn, metric):
    '''
    This function replaces text found in the file name.
    '''
    # Text contained within the input file names are replaced to be more concise.
    # This is done to shorten the length of the plot's name
    text = fn.replace("target-byte", "tb")
    # Additional text is replaced if the metric is TVLA
    if metric == "TVLA":
        text = text.replace("byte-value", "bv")
    return text