import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
# model = tf.keras.models.load_model('Saved Models/8_cat/4_7_2023_86p(80_20_split).h5')
model = tf.keras.models.load_model('Saved Models/8_cat/mobilenet_84p(90_10_split).h5')

# Load the class labels
class_labels = ['Duracell', 'Energizer', 'Eveready', 'Exell', 'GP', 'Ikea', 'Klarus', 'Panasonic']

# Load the class labels
# class_labels = ['Duracell', 'Energizer', 'Eveready', 'Exell', 'GP', 'Ikea', 'Klarus', 'Not Batteries', 'Panasonic', 'Unknown/Other Brands']

input_height = 224
input_width = 224
# Set the input size expected by your model
input_size = (input_height, input_width) 

# Open a video capture stream using the USB camera
width, height = 320, 240
framerate = 30
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, framerate)

while True:
    # Read frame from the video stream
    ret, frame = cap.read()

    # Perform object detection using Keras model
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = frame_resized.astype('float') / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    predictions = model.predict(frame_expanded)
    
    # Process the predictions
    for prediction in predictions:
        # Get the class label index with the highest confidence
        class_index = np.argmax(prediction)
        class_label = class_labels[class_index]

        # Get the confidence score of the predicted class
        confidence = prediction[class_index]

        # Display the class label and confidence
        if confidence >0.7:
            label = f'{class_label}: {confidence:.2f}'
        else:
            label = 'detecting...'
        
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Display the output frame
    cv2.imshow('Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture stream and close all windows
cap.release()
cv2.destroyAllWindows()
