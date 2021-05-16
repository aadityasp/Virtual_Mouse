import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
cap = cv2.VideoCapture(0)

width,height = 640,480
cap.set(3,width)
cap.set(4,height)
while True:
    success ,img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)