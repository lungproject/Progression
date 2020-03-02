from __future__ import print_function

import datetime
import keras
import numpy as np
import tensorflow as tf
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import  Sequential
from keras.layers import Input,Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold

from keras import backend as K

from Loaddata_all import load_alldata
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def binary_focal_loss(gamma=2., alpha=.25):
    
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

model = load_model('./model/ncppredictionmodel.hdf5')

x_train ,y_train ,x_test, y_test  = load_alldata()#load_data(count)

predicttest = model.predict(x_test, verbose=1)#ADCdata2,,xfuse_test
np.savetxt("./Results/predicttest.txt",predicttest)

predicttrain = model.predict(x_train, verbose=1)#ADCdata1,,xfuse_train
np.savetxt("./Results/predicttrain.txt",predicttrain)




# model2 = Model(inputs=model.input,
#                 outputs=model.get_layer('activation_17').output)#创建的新模型

# activations = model2.predict(x_train)#W
# print(activations.shape)
# np.save('activations17train.npy', activations)


# model2 = Model(inputs=model.input,
#                 outputs=model.get_layer('global_average_pooling2d_1').output)#创建的新模型

# activations = model2.predict(x_train)#W
# print(activations.shape)
# np.save('pool17train.npy', activations)


# model2 = Model(inputs=model.input,
#                 outputs=model.get_layer('activation_7').output)#创建的新模型

# activations = model2.predict(x_train)#W
# print(activations.shape)
# np.save('activations7train.npy', activations)


# model2 = Model(inputs=model.input,
#                 outputs=model.get_layer('activation_17').output)#创建的新模型

# activations = model2.predict(x_test)#W
# print(activations.shape)
# np.save('activations17test.npy', activations)

# model2 = Model(inputs=model.input,
#                 outputs=model.get_layer('global_average_pooling2d_1').output)#创建的新模型

# activations = model2.predict(x_test)#W
# print(activations.shape)
# np.save('pool17test.npy', activations)


# model2 = Model(inputs=model.input,
#                 outputs=model.get_layer('activation_5').output)#创建的新模型

# activations = model2.predict(x_test)#W
# print(activations.shape)
# np.save('activations7test.npy', activations)


