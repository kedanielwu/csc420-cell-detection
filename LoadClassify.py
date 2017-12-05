import os
import imageio
import numpy as np
import cv2


def load_class_data():
    epithelial_path = '../epithelial'
    fibroblast_path = '../fibroblast'
    inflammatory_path = '../inflammatory'
    others_path = '../others'
    epithelial_example = os.listdir(epithelial_path)
    fibroblast_example = os.listdir(fibroblast_path)
    inflammatory_example = os.listdir(inflammatory_path)
    others_example = os.listdir(others_path)


    # epithelial
    epithelial = [imageio.imread('../epithelial/' + epithelial_example[i]) for i in range(1,7001)]
    epithelial_label = np.zeros((7000, 1), dtype=np.int)

    # fibroblast
    fibroblast = [imageio.imread('../fibroblast/' + fibroblast_example[i]) for i in range(1, 4001)]
    fibroblast_label = np.ones((7000, 1), dtype=np.int)

    # inflammatory
    inflammatory = [imageio.imread('../inflammatory/' + inflammatory_example[i]) for i in range(1, 6001)]
    inflammatory_label = np.full((6000, 1), 2, dtype=np.int)

    # others
    others = [imageio.imread('../others/' + others_example[i]) for i in range(1, 1501)]
    others_label = np.full((1500, 1), 3, dtype=np.int)

    result = fibroblast + epithelial + inflammatory + others

    labels = np.concatenate((fibroblast_label, epithelial_label, inflammatory_label, others_label))
    meta = np.array([(result[j],labels[j]) for j in range(len(result))])
    print('waiting shuffle\n')
    print(len(meta))
    np.random.shuffle(meta)
    res = np.concatenate([arr[0][np.newaxis] for arr in meta])
    label_r = np.concatenate([arr[1][np.newaxis] for arr in meta])
    print(label_r)

    # epithelial_t
    epithelial_t = [imageio.imread('../epithelial/' + epithelial_example[i]) for i in range(6001, 6101)]
    epithelial_label_t = np.zeros((100, 1), dtype=np.int)

    # fibroblast_t
    fibroblast_t = [imageio.imread('../fibroblast/' + fibroblast_example[i]) for i in range(4001, 4101)]
    fibroblast_label_t = np.ones((100, 1), dtype=np.int)

    # inflammatory_t
    inflammatory_t = [imageio.imread('../inflammatory/' + inflammatory_example[i]) for i in range(5001, 5101)]
    inflammatory_label_t = np.full((100, 1), 2, dtype=np.int)

    # others_t
    others_t = [imageio.imread('../others/' + others_example[i]) for i in range(1801, 1901)]
    others_label_t = np.full((100, 1), 3, dtype=np.int)

    result_t = epithelial_t + fibroblast_t + inflammatory_t + others_t
    labels_t = np.concatenate(
        (epithelial_label_t, fibroblast_label_t, inflammatory_label_t, others_label_t))
    res_t = np.concatenate([arr[np.newaxis] for arr in result_t])

    return (res, label_r), (res_t, labels_t)



