import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

cap = cv2.VideoCapture(0)

width, height = 640, 480
cap.set(3, width)
cap.set(4, height)
detector = htm.handDetector(maxHands=1)
print(detector.maxHands)
xmin, xmax = int(width / 8), int(7 * width / 8)
ymin, ymax = int(height / 4), int(3 * height / 4)
bbox_default = xmin, ymin, xmax, ymax


# intersection = 0
def calculateIntersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1:  # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1:  # Contains
        intersection = b1 - b0
    elif a0 < b0 < a1:  # Intersects right
        intersection = a1 - b0
    elif a1 > b1 > a0:  # Intersects left
        intersection = b1 - a0
    else:  # No intersection (either side)
        intersection = 0

    return intersection


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # detector = htm.handDetector()
    img = detector.findHands(img)
    lmlst, bbox = detector.findPosition(img)
    if bbox:
        X0, Y0, X1, Y1, = bbox
        AREA = float((X1 - X0) * (Y1 - Y0))
        rectangle = [bbox_default]
        # intersecting=[]
        for x0, y0, x1, y1 in rectangle:
            width = calculateIntersection(x0, x1, X0, X1)
            height = calculateIntersection(y0, y1, Y0, Y1)
            area = width * height
            percent = area / AREA
            # if percent >= 0.2:
            #     intersecting.append([x0, y0, x1, y1])
            if percent >= 0.2:
                print("IN Active Range")
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            else:
                print("NOT IN Active Range")
        # print( percent, intersecting)
        # if not ((bbox[0] > bbox_default[2] or bbox[2] < bbox_default[0]) and (
        #         bbox[1] > bbox_default[3] or bbox[3] < bbox_default[1])):
        #     print("IN Active Range")
        # else:
        #     print("NOT IN Active Range")

            # img = detector.findHands(img)
    # cv2.rectangle(img, (xmin-20 , ymin ), (xmax +20 , ymax ), (0, 0,255), 2) #active region
    cv2.imshow("Image", img)
    cv2.waitKey(1)
