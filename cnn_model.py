# Importing the Keras libraries and packages
import tensorflow as tf
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Dense , Dropout
from keras.layers import Flatten
import os
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# Specifying that we are using GPU while training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize the CNN model
model = Sequential()

# 1st Convolutional layer
model.add(Convolution2D(32,(3,3), input_shape = (300,300,3), activation = 'relu'))

# 1st Maxpool layer
model.add(MaxPooling2D((2,2)))

# 2nd Convolutional layer
model.add(Convolution2D(64, (3,3), activation = 'relu'))

# 2nd Maxpool layer
model.add(MaxPooling2D((2,2)))

# 3rd Convolutional layer
model.add(Convolution2D(64, (3,3), activation = 'relu'))

# Adding Flatten layer
model.add(Flatten())

# Input layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(96, activation='relu'))
model.add(Dropout(0.40))

model.add(Dense(64, activation='relu'))

# Here 27 is the number of category, right now it is not fixed
model.add(Dense(36, activation='softmax'))

# Compiling the CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2 categories

# print(model.summary())


training_datagen = ImageDataGenerator(rescale=1./255, shear_range= 0.2, zoom_range=0.2)

testing_datagen = ImageDataGenerator(rescale=1./255)


training_dataset = training_datagen.flow_from_directory('dataset/train_set', target_size= (300,300), batch_size=20, color_mode='rgb', class_mode= 'categorical')

testing_dataset = testing_datagen.flow_from_directory('dataset/test_set', target_size= (300,300), batch_size=20, color_mode='rgb', class_mode= 'categorical')

batch_size = 50

model.fit_generator(training_dataset, steps_per_epoch= 43681 // 20, epochs = 8, validation_data= testing_dataset, validation_steps= 13000 // 20)

# Saving the model
saved_model = model.to_json()

with open('sign_model.json','w') as json_file:
    json_file.write(saved_model)
print("Model Successfully Saved")

# Saving the model's weight
model.save_weights('sign_model.h5')
print('Weight Successfully Saved')




