import cv2
import numpy as np
import math
import time
import tkinter as tk
from tkinter import Text, Button, Scrollbar  # Import Scrollbar widget
from tensorflow import keras
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Create a Tkinter window
root = tk.Tk()
root.title("Sign Detection")
root.geometry("800x600")

# Create a Scrollbar
scrollbar = Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a textarea to display the composed sentence and configure it to use the scrollbar
text_area = Text(root, font=("Helvetica", 20), yscrollcommand=scrollbar.set)
text_area.pack(pady=20)

# Configure the scrollbar to work with the text_area
scrollbar.config(command=text_area.yview)

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM)

# Create a Send button with a color


def send_text():
    composed_sentence = text_area.get("1.0", tk.END)


    # You can add code here to send the composed sentence to your desired destination
send_button = Button(button_frame, text="Send", command=send_text, font=(
    "Helvetica", 16), relief=tk.RAISED, borderwidth=3, bg="green", fg="white")
send_button.pack(side=tk.LEFT, padx=10)


def discard_text():
    text_area.delete("1.0", tk.END)  # Clears the textarea


discard_button = Button(button_frame, text="Discard", command=discard_text, font=(
    "Helvetica", 16), relief=tk.RAISED, borderwidth=3, bg="red", fg="white")
discard_button.pack(side=tk.LEFT, padx=10)

detector = HandDetector(maxHands=1)
classifier = Classifier('model/keras_model_v6.h5', 'model/labels.txt')
counter = 0
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
sentence = ""
last_detection_time = time.time()
detection_interval = 3  # interval for detection in seconds

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    # Calculate the time elapsed since the last detection
    current_time = time.time()
    time_elapsed = current_time - last_detection_time

    if hands and time_elapsed >= detection_interval:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset: x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Add the detected sign to the sentence
        detected_sign = labels[index]
        sentence += detected_sign

        # Update the textarea with the composed sentence in sentence case
        text_area.delete("1.0", tk.END)  # Clear previous text
        # Display composed sentence in sentence case
        text_area.insert(tk.END, sentence.capitalize())

        # Display the detected sign on the OpenCV window in sentence case
        cv2.putText(imgOutput, detected_sign, (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

        # Update the last detection time
        last_detection_time = current_time

    cv2.imshow("Image", imgOutput)

    # Check for the "Escape" key press to exit the program
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the "Escape" key
        break

    # Update the Tkinter window to handle GUI events
    root.update()

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the Tkinter window
root.destroy()
