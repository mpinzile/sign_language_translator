import cv2
import numpy as np
import math
from tensorflow import keras
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

detector = HandDetector(maxHands=1)
classfier = Classifier('model/keras_model_v6.h5', 'model/labels.txt')
counter = 0
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset: x + w + offset]

        # Check if imgCrop is empty before resizing
        if not imgCrop.size == 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction, index = classfier.getPrediction(
                    imgWhite, draw=False)
            else:
                k = imgSize/w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal)/2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction, index = classfier.getPrediction(
                    imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x-offset, y-offset-50),
                          (x - offset + 90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-30),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                          (x + w+offset, y + h+offset), (255, 0, 255), 4)

    cv2.imshow("Image", imgOutput)

    # Check for the escape key (ASCII code 27)
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the escape key
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
