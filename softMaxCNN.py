from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import scipy.io as sio
from LoadClassify import load_class_data

batch_size = 50
epochs = 50
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'class_cnn_model'

(data, label), (data_t, label_t) = load_class_data()
print(data.shape)
print(label.shape)
print(data_t.shape)
print(label_t.shape)

label = keras.utils.to_categorical(label,4)
label_t = keras.utils.to_categorical(label_t,4)

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
model.add(Dense(4))
model.add(Activation('softmax'))

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=180,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=True,
    vertical_flip=False
)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

data = data.astype('float32')
data_t = data_t.astype('float32')

data /= 255
print(data)

datagen.fit(data)
model.fit_generator(datagen.flow(data, label,batch_size=50), epochs=epochs, validation_data=(data_t, label_t), workers=1, steps_per_epoch=500,shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

