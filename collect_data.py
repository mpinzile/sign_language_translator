import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(maxHands=1)
character = 'A'
folder = f'images/{character}'
counter = 0
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset: x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Check if imgCrop is valid
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w) if h > 0 else 0
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h) if w > 0 else 0
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("Cropped Image", imgCrop)
            cv2.imshow("White Image", imgWhite)

            key = cv2.waitKey(1)
            if key == ord('s') or key == ord('S'):
                counter += 1
                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                print(counter)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
