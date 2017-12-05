import os
import imageio
import numpy as np


def load_cell_data():
    positive_path = '../cell'
    negative_path = '../negative'
    positive_example = os.listdir(positive_path)
    negative_example = os.listdir(negative_path)

    # training 1
    print('reading positive examples\n')
    positive_arr1 = [imageio.imread('../cell/' + positive_example[i]) for i in range(1,5001)]
    positive_label1 = np.ones((5000, 1), dtype=np.int)
    print('reading negative examples\n')
    negative_arr1 = [imageio.imread('../negative/' + negative_example[i]) for i in range(1, 5001)]
    negative_label1 = np.zeros((5000, 1), dtype=np.int)

    # test
    print('reading positive examples\n')
    positive_arr_t = [imageio.imread('../cell/' + positive_example[i]) for i in range(50001, 55001)]
    positive_label_t = np.ones((5000, 1), dtype=np.int)
    print('reading negative examples\n')
    negative_arr_t = [imageio.imread('../negative/' + negative_example[i]) for i in range(10001, 15001)]
    negative_label_t = np.zeros((5000, 1), dtype=np.int)

    result = positive_arr1 + negative_arr1
    labels = np.concatenate((positive_label1, negative_label1))
    res = np.concatenate([arr[np.newaxis] for arr in result])

    result_t = positive_arr_t + negative_arr_t
    labels_t = np.concatenate(
        (positive_label_t, negative_label_t))
    res_t = np.concatenate([arr[np.newaxis] for arr in result_t])

    return (res, labels), (res_t, labels_t)



