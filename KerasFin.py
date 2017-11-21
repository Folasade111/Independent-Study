from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
import numpy
from keras.preprocessing.image import ImageDataGenerator
#dimensions of image
img_width,img_height = 150,150
train_data_dir = 'genderdata/Trainer'
validation_data_dir = 'genderdata/Validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 30
batch_size = 16
input_shape = (img_width, img_height,3)

model = Sequential()
#creating 3 stacks of convolution layers
#stack 1
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

#the threee models have been outputing 3d features map(height,width,features)

model.add(Flatten()) #this converts 3d features to 1d features
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics =['accuracy'])


train_datagen =ImageDataGenerator(rescale=1. /255,
                                  shear_range = 0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size)
model.save_weights('gender_try.h5')
