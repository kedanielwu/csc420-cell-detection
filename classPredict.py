from keras.models import load_model
import os
import imageio
import numpy as np
import scipy.io as sp
import keras
from skimage.util.shape import view_as_blocks
from PIL import Image


def load_test_data(number):

    im = imageio.imread('../data/Detection/img{}/img{}.bmp'.format(str(number), str(number)))
    detection = sp.loadmat('../data/Detection/img{}/img{}_detection.mat'.format(str(number), str(number)))
    block_size = 27
    counter = 0

    data = np.zeros((1,27,27,3))
    for i in detection['detection']:
        x1 = int(max(0, i[0] - 14))
        x2 = int(min(499, i[0] + 12))
        y1 = int(max(0, i[1] - 14))
        y2 = int(min(499, i[1] + 12))

        if x1 == 0:
            x2 = 26
        if x2 == 499:
            x1 = 473
        if y1 == 0:
            y2 = 26
        if y2 == 499:
            y1 = 473

        print('{}, {}, {}, {}'.format(x1,x2,y1,y2))
        imageio.imwrite('{}.bmp'.format(str(counter)), im[y1:y2+1, x1:x2+1, :])
        counter += 1
        data = np.concatenate((data, im[y1:y2+1, x1:x2+1, :][np.newaxis]))

    return data


model = load_model('./saved_models/class_cnn_model')

for k in range(1,2):
    test_data = load_test_data(k)
    print(test_data.shape)
    res = model.predict_proba(test_data[1:], batch_size=32)
    print(res)
    sp.savemat('./img{}_class.mat'.format(str(k)), mdict={'arr': res})
