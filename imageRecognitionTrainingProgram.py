# python script that takes in database of images for traing and creates a model of what a vehicle is 'supposed' to look like 
# time spent training is dependent upon how many image files are ran through the program.

# @chasealbright


from keras import applications
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Convolution2D, MaxPooling2D, Activation
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.models import load_model
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import keras
from urllib.request import urlopen
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

inputImageSize = [1280, 720]

def create_model():   

    nb_conv = 5
    nb_filters = 8
    model = Sequential()
    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv),
                            border_mode='valid',
                            input_shape=image_size, dim_ordering="tf") ) 
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv)))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(nb_filters*2, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model

def train_model(batch_size = 50, n_epoch = 20):
    train_data, train_target = read_and_normalize_train_data()
    
    cv_size = int(train_target.size/4)

    train_data[:,:,:] = train_data[0:,:,:]
    train_target = train_target[0:]

    X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=cv_size, random_state=56741)

    model = create_model()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(X_valid, y_valid) )

    predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
    compare = pd.DataFrame(data={'original':y_valid.reshape((cv_size,)),
             'prediction':predictions_valid.reshape((cv_size,))})
    compare.to_csv('compare.csv')
    
    return model

def load_train():
    x_train = []
    y_train = []

    datas = loadTrainingData(foldername)
    print('Read training images')
    i=0
    for row in datas:
        image_path = row[0]
        try:
            resp = urlopen(image_path)
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.cvtColor(cv2.imdecode(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (image_size_c, image_size_r))
            x_train.append(img)
            y_train.append(int(row[1]))   
            i=i+1
            if i>6000:
                break
        except:
            print(image_path)
        
         
    
    print('Training images read')
    return (np.expand_dims(x_train, axis=3), y_train)

def loadTrainingData(foldername):

    fullName = foldername + r'/freemontCorrectedResults.csv'
    file = open(fullName)
    datas = []
    for line in file:
        try:
            data = line.split(",")
            datas.append(data)
        except:
            print("")
    numel = len(datas)
    return datas

def read_and_normalize_train_data():
    train_data, train_target = load_train()
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

keras.backend.clear_session()
image_width = inputImageSize[0]
image_height = inputImageSize[1]
keras.backend.clear_session()
imageDownsizeFactor = 2
image_size_r = int(image_height/2) 
image_size_c = int(image_width/2)

image_size = (image_size_r, image_size_c, 1)
foldername = r'//share/FreemontTraining'
model = train_model(n_epoch = 60)
model.save('FreemontModelsavetest.h5')

model = load_model('FreemontModelsavetest.h5')

