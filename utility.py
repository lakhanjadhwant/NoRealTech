import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import layers, models
from keras.models import load_model
import numpy as np
import cv2
import numpy as np
import os
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
# Create a simple CNN model
from keras import models, layers


def capture_samples(name):
    # Directory to save captured photos
    save_dir = 'Train'
    
    save_dir = save_dir + "/" + name
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Counter to keep track of captured photos
    photos_count = 0
    
    while photos_count < 600:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
    
        # If a face is detected, save the photo
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
                # Save the detected face
                face_img = frame[y:y+h, x:x+w]
                img_name = os.path.join(save_dir, f"photo_{photos_count}.jpg")
                cv2.imwrite(img_name, face_img)
    
                print(f"Photo {photos_count + 1} captured and saved as {img_name}")
    
                # Increment photo count
                photos_count += 1
    
        # Display the resulting frame
        cv2.imshow('Capture Photos', frame)
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
def Train_data():
    main_directory = "."

    # Specify the training dataset directory
    train_directory = main_directory + "/Train"

    image_size = (256, 256)
    batch_size = 32

    # Define the main directory containing the 'train' and 'test' folders
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training',
       
    )

    return train_dataset,train_dataset.class_names

    

def model_train(train_dataset,output):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_dataset,
        epochs=2, 
    )
    return model

def face_detect(model,class_data):                                                                                                                                                                                                                                      
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    font = cv2.FONT_HERSHEY_COMPLEX

    def get_className(classNo,class_data):
        return class_data[classNo]

    while True:
        success, imgOrignal = cap.read()
        faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
        for x, y, w, h in faces:
            crop_img = imgOrignal[y:y+h, x:x+h]
            img = cv2.resize(crop_img, (256, 256))
            img = img.reshape(1, 256, 256, 3)
            prediction = model.predict(img)
            classIndex = np.argmax(prediction)
            probabilityValue = np.amax(prediction)

            cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y), (x+w, y), (0, 255, 0), 2)
            cv2.putText(imgOrignal, str(get_className(classIndex,class_data)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(imgOrignal, str(round(probabilityValue*100, 2))+"%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()