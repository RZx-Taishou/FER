import tensorflow as tf
import cv2
import numpy as np

new_model=tf.keras.models.load_model('FER_model.h5')

# Set the path for the face cascade classifier
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(face_cascade_path)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Resize and preprocess the face image for the model
        final_image = cv2.resize(roi_color, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        # Make predictions using the model
        Predictions = new_model.predict(final_image, verbose=0)

        # Get the emotion label
        emotion_label = np.argmax(Predictions)

        # Define the emotion status
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        status = emotions[emotion_label]

        # Display the emotion status on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, status, (x, y - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Emotion Recognition", frame)

    # Break the loop if 'q' or 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' key or 'Esc' key
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()