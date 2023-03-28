# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

# Set parameters
batch_size = 32
epochs = 10
frames = 16
rows = 112
cols = 112
channels = 3
num_classes = 10

# Set up data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(rows, cols),
        batch_size=batch_size,
        class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(rows, cols),
        batch_size=batch_size,
        class_mode='categorical')

# Define the CNN architecture
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(frames, rows, cols, channels)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          epochs=epochs,
          validation_data=val_generator)

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(rows, cols),
        batch_size=batch_size,
        class_mode='categorical')
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
