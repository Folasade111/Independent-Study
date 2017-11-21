#from __future__ import print_function
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.models import Sequential
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np
import os
from numpy import array
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
model = Sequential()
dire = '/home/afolasade/PycharmProjects/ML/Testing/'
folder = os.listdir(dire)


img_width,img_height = 150,150
input_shape = (img_width, img_height, 3)
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#stack 2

model.add(Conv2D(32,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#stack 3

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #this converts 3d features to 1d features
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.load_weights('gender_try.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
things_to_classify = []
#datafile = open('filey.txt','w+')
#male = 1
#female = 0
data = 'IMG_5681.JPG'
#print >> datafile, data
img = Image.open(dire+data)
img = img.resize((150, 150))
things_to_classify.append(np.asarray(img, dtype=np.uint8))


arr = array(things_to_classify)
#desicionfile = open('file1.txt','w+')
#print Image
classes = model.predict_classes(arr,batch_size=32,verbose=0)
for cl in classes:
    if cl[0] == 0:
        print 'female'
    else:
        print 'male'

