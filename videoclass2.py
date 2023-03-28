import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the CNN model
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Load the data and preprocess it
train_data = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_dataset = train_data.flow_from_directory(
    directory="dataset/train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
)

validation_data = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
validation_dataset = validation_data.flow_from_directory(
    directory="dataset/validation",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=validation_dataset,
)

# Evaluate the model on the test data
test_data = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_dataset = test_data.flow_from_directory(
    directory="dataset/test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=False,
)

model.evaluate(test_dataset)



# Save the model
model.save('dataset/parvesh.h5')
