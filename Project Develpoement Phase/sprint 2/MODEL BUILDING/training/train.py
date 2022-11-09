import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
x_train = train_datagen.flow_from_directory('/home/siva/Documents/bro/Nutri/Dataset/TRAIN_SET',target_size=(64,64),batch_size=32,class_mode='binary')
x_test = test_datagen.flow_from_directory('/home/siva/Documents/bro/Nutri/Dataset/TEST_SET',target_size=(64,64),batch_size=32,class_mode='binary')
print(x_train.class_indices)

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=5,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(generator= x_train, steps_per_epoch= len(x_train), epochs=20, validation_data=x_test, validation_steps=len(x_test))
model.save('nutrition.h5')

