# Importing lib
import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("keras_model.h5")

# Open the camera
camera = cv2.VideoCapture(0)

# Mapping class indices to variables or labels
class_mapping = {
    0: "Rock",
    1: "Paper",
    2: "Scissors"
}

#Loop that checks the frames
while True:
    # Read a frame from the camera
    status, frame = camera.read()

    # If successful in reading the frame we start predicting
    if status:
        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # Resize the frame to (224, 224)
        frame = cv2.resize(frame, (224, 224))

        # Expand dimensions and normalize the frame
        expanded_frame = np.expand_dims(frame, axis=0).astype(np.float32) / 255.0

        # Get predictions from the model
        predictions = model.predict(expanded_frame)

        # Extract the predicted class
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Get the variable or label associated with the predicted class
        predicted_variable = class_mapping.get(predicted_class, "Unknown")

        print("Predicted class:", predicted_variable)

        # Display the captured frame
        cv2.imshow('Frames_Captured', frame)

        # Wait for 1ms and check if the spacebar is pressed
        code = cv2.waitKey(1)
        if code == 32:  # ASCII code for spacebar
            # Press spacebar to leave
            break

# Release the camera
camera.release()

# Close the open window
cv2.destroyAllWindows()