import cv2
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('C:/Users/rames/Desktop/Video-Classifier-Using-CNN-and-RNN - Copy/dataset/parvesh.h5')

# Define a function to preprocess the image
def preprocess_image(frame):
    # Convert the color space from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize the image to 224x224 (the input shape of the model)
    image = cv2.resize(image, (224, 224))
    # Normalize the image
    image = image.astype("float32") / 255.0
    # Add an extra dimension to the image to represent the batch size (1 in this case)
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to detect crime from a frame
def detect_crime(frame, model):
    # Preprocess the image
    image = preprocess_image(frame)
    # Make a prediction using the model
    prediction = model.predict(image)
    # Return the prediction
    return prediction[0][0]

# Initialize the video capture device
cap = cv2.VideoCapture("C:/Users/rames/Downloads/violence video classifier/Real Life Violence Dataset/Violence/V_998.mp4")

# Loop through the frames from the webcam
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    # Detect crime from the frame
    prediction = detect_crime(frame, model)
    # Display the frame and the prediction
    cv2.imshow('frame', frame)
    if prediction <= 0.5:
        print('Crime detected!')
    else:
        print('No crime detected')
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()