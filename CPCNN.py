from __future__ import print_function
import keras
from sklearn.metrics import precision_recall_curve
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import scipy.io as sio
from LoadImg import load_cell_data

class Metrics(keras.callbacks.Callback):
    def __init__(self):
        self.precision_recall = np.array([])
    def on_epoch_end(self, batch, logs={}):
        correct = 0
        gt = 0
        pd = 0
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        for i in range(predict.shape[0]):
            if targ[i][1] == 1:
                gt += 1
            if predict[i][1] > predict[i][0]:
                pd += 1
            if predict[i][1] > predict[i][0] and targ[i][1] == 1:
                correct += 1
        print('\n', gt, '\n', pd, '\n', correct, '\n')
        precision = correct / pd
        recall = correct / gt
        res = np.array([precision, recall])
        self.precision_recall=np.concatenate((self.precision_recall, res))
        print("percision: ", precision, "\n")
        print("recall: ", recall, "\n")
        return

    def on_train_end(self, logs=None):
        sio.savemat('./precision_recall.mat', mdict={'pr': self.precision_recall})

pr = Metrics()
batch_size = 50
num_classes = 2
epochs = 50
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cell_cnn_model'

(data, label), (data_t, label_t) = load_cell_data()

label = keras.utils.to_categorical(label,2)
label_t = keras.utils.to_categorical(label_t,2)

model = Sequential()
model.add(Conv2D(36, (4, 4), padding='same',
                 input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

data = data.astype('float32')
data_t = data_t.astype('float32')

data /= 255
data_t /= 255

model.fit(data,
          label,
          batch_size=batch_size, epochs=epochs,
          validation_data=(data_t, label_t),
          shuffle=True,
          callbacks=[pr])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

