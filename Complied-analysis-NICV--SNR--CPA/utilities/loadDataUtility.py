"""
This script contains functions required for loading and preparing the data required for training the model.
"""

import os

import numpy as np
import pandas as pd

sbox = [
    # 0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,  # 0
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,  # 1
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,  # 2
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,  # 3
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,  # 4
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,  # 5
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,  # 6
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,  # 7
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,  # 8
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,  # 9
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,  # a
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,  # b
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,  # c
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,  # d
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,  # e
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16  # f
]


def list_files(data_path):
    """
    This functions lists the files in the directory.
    """
    files = os.listdir(data_path)
    print('Total number of files in the directory: ', len(files))
    print('files are: %s' % files)

    return files


def load_data(data_path):
    """
    This function loads the data from the path and returns the numpy array containing the data.
    """
    # files = list_files(data_path)
    # print('files are as follows: ', files)
    # reading the files
    train_data_path = os.path.join(data_path, 'train_same_key.npz')
    test_data_path = os.path.join(data_path, 'test_same_key.npz')
    test_data_diff_key_path = os.path.join(data_path, 'test_diff_key.npz')

    print('loading the dataset ...')
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    test_data_diff_key = np.load(test_data_diff_key_path)
    print('data loaded successfully!')

    return train_data, test_data, test_data_diff_key


def data_info(data_matrix):
    """
    This function prints the information of the dataset loaded. The function prints the information for training,
    and testing data with same and different key.
    """
    plain_text, power_traces, key = data_matrix['plain_text'], data_matrix['power_trace'], data_matrix['key']

    print('shape of the plain text matrix : ', plain_text.shape)
    print('shape of the power trace matrix: ', power_traces.shape)
    print('Encryption key: ', key)
    print('-' * 90)


def aes_internal(inp_data_byte, key_byte):
    """
    This function performs XOR operation between the input byte and key byte which is used as label.
    """
    return sbox[inp_data_byte ^ key_byte]


def gen_features_labels(data, input_key_byte, start_index, end_index):
    """
    This function generates the labels and features pair for training a model.
    The labels are generated based on XOR operation between the plaintext byte and the key byte.
    """
    
    data_files = data.files
    # print('data files: ', data_files)
    
    # if files contained in the zip are power_trace, plain_text, and key
    if 'power_trace' in data_files:
        # print('loading Jimmy\'s dataset ...')
        # for loading my dataset
        power_traces, plain_text, key = data['power_trace'], data['plain_text'], data['key']
    else:
        # print('loading chenggang\'s dataset ...')
        # if using data from chenggang's dataset
        power_traces, plain_text, key = data['trace_mat'], data['textin_mat'], data['key']
        
            
    
    
    # for loading my dataset
    # power_traces, plain_text, key = data['power_trace'], data['plain_text'], data['key']

    # if using data from chenggang's dataset
    # power_traces, plain_text, key = data['trace_mat'], data['textin_mat'], data['key']
    # key byte is value between 0 to 15
    labels = []
    for i in range(plain_text.shape[0]):
        text_i = plain_text[i]
        label = aes_internal(text_i[input_key_byte], key[input_key_byte])
        labels.append(label)

    labels = np.array(labels)
    if not isinstance(power_traces, np.ndarray):
        power_traces = np.array(power_traces)
    power_traces = power_traces[:, start_index:end_index]
    return power_traces, labels, plain_text, key


def save_gen_features_labels(params, train_data, test_data_k1, test_data_k2):
    """
    This function saves the data with traces, and appropriate labels, which is later used for training a model.
    """
    target_byte = params["key_byte"]
    print('processing training data ...')
    power_traces, labels, plain_text, key = gen_features_labels(train_data,
                                                                params["key_byte"],
                                                                params["start_index"],
                                                                params["end_index"])
    print('training data processing completed!')
    print('processing testing data with same key as training data ...')
    power_traces_test_k1, labels_test_k1, plain_text_k1, key_k1 = gen_features_labels(test_data_k1, params["key_byte"],
                                                                                      params["start_index"],
                                                                                      params["end_index"])
    print('processing completed for same key as training data!')
    print('processing testing data with different key ...')
    power_traces_test_k2, labels_test_k2, plain_text_k2, key_k2 = gen_features_labels(test_data_k2, params["key_byte"],
                                                                                      params["start_index"],
                                                                                      params["end_index"])
    print('processing testing data with different key completed!')
    # saving training data after preprocessing
    data_path_dir = os.path.join(params["data_path"], 'processed_data_byte_{}'.format(target_byte))
    if not os.path.isdir(data_path_dir):
        os.makedirs(data_path_dir)

    train_data_path = os.path.join(data_path_dir, 'train_data_same_key.npz')
    test_data_k1 = os.path.join(data_path_dir, 'test_data_same_key.npz')
    test_data_k2 = os.path.join(data_path_dir, 'test_data_diff_key.npz')

    # saving the data
    np.savez(train_data_path, data=power_traces, label=labels, plain_text=plain_text, key=key)
    np.savez(test_data_k1, data=power_traces_test_k1, label=labels_test_k1, plain_text=plain_text_k1, key=key_k1)
    np.savez(test_data_k2, data=power_traces_test_k2, label=labels_test_k2, plain_text=plain_text_k2, key=key_k2)

    # looking at information of preprocessed dataset
    print('Training dataset: ')
    print('features shape: ', power_traces.shape)
    print('labels shape  : ', labels.shape)
    print('unique labels : ', len(np.unique(labels)))
    print('-' * 90)
    print('Testing dataset with key k1: ')
    print('features shape: ', power_traces_test_k1.shape)
    print('labels shape  : ', labels_test_k1.shape)
    print('unique labels : ', len(np.unique(labels_test_k1)))
    print('-' * 90)
    print('Testing dataset with key k2: ')
    print('features shape: ', power_traces_test_k2.shape)
    print('labels shape  : ', labels_test_k2.shape)
    print('unique labels : ', len(np.unique(labels_test_k2)))


def create_df(power_traces, power_traces_labels, n):
    """
    This function creates a dataframe from the numpy array and generates the subset of the dataset which is used for
    training the feature extractor
    :param power_traces: The power traces used for training the model
    :param power_traces_labels: The labels corresponding to the power traces
    :param n: Number of traces to be selected for each class
    :return: the subset of the dataset
    """
    # creating dataframe of the data for selecting subset of the dataset
    y_df = pd.DataFrame(data=power_traces_labels, columns=['label'])
    x_df = pd.DataFrame(data=power_traces)
    frames = [y_df, x_df]
    # creating the dataframe
    all_data = pd.concat(frames, axis=1)
    all_data = all_data.sort_values(by=['label'])

    # select N samples from each class
    grouped = all_data.groupby(['label'], as_index=False)
    all_data = grouped.apply(lambda frame: frame.sample(n))
    all_data.index = all_data.index.droplevel(0)
    all_data = all_data.reset_index(drop=True)

    # separating power traces and labels
    power_traces = all_data.iloc[:, 1:]
    power_traces = power_traces.to_numpy()
    power_traces_label = all_data['label']
    print('shape of the power traces to be used for training: ', power_traces.shape)
    return [power_traces, power_traces_label, all_data]
