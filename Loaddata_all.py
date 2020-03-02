import os  
from PIL import Image  
import csv
import numpy as np  
from keras import backend as K
import scipy.io
from scipy.io import loadmat


def load_alldata(): #ct window
     
    img = np.load("./alldata/smallxptrain_128.npy") #Deeplearningallpatchsmalls  3part3
    datatrain = np.asarray(img,dtype="float32")

    
    img = np.load("./alldata/smallyptrain.npy")
    labeltrain = np.asarray(img,dtype="float32")


    img = np.load("./alldata/smallxptest_128.npy") #Deeplearningallpatchsmalls  3part3
    datatest = np.asarray(img,dtype="float32")

    
    img = np.load("./alldata/smallyptest.npy")
    labeltest = np.asarray(img,dtype="float32")
    
    datatrain = np.expand_dims(datatrain, axis=3)
    datatest = np.expand_dims(datatest , axis=3)

    return datatrain,labeltrain,datatest,labeltest


def load_yidadata(): #ct window
     
    img = np.load("./alldata/yidadata_128.npy") #Deeplearningallpatchsmalls  3part3
    data = np.asarray(img,dtype="float32")

    
    img = np.load("./alldata/yidalabels.npy")
    label = np.asarray(img,dtype="float32")


    
    
    data= np.expand_dims(data, axis=3)
 

    return data,label 
  
