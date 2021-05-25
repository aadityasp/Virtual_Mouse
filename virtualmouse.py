import cv2
import numpy as np
import HandTracking as htm
import time
import autopy

cap = cv2.VideoCapture(0)

c_width, c_height = 640, 480
cap.set(3, c_width)
cap.set(4, c_height)
detector = htm.handDetector(maxHands=2)
# print(detector.maxHands)
xmin, xmax = int(c_width / 8), int(7 * c_width / 8)
ymin, ymax = int(c_height / 4), int(3 * c_height / 4)
bbox_default = xmin, ymin, xmax, ymax
# print("BBox[0]",bbox_default[0])
screen_width, screen_height = autopy.screen.size()
print(screen_width, screen_height)
click_flag = 0
smooth_factor = 5
# intersection = 0

x_before_smoothing, y_before_smoothing = 0, 0
x_after_smoothing, y_after_smoothing = 0, 0


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
                # print("IN Active Range")
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            # else:
            # print("NOT IN Active Range")
        # print( percent, intersecting)
        # if not ((bbox[0] > bbox_default[2] or bbox[2] < bbox_default[0]) and (
        #         bbox[1] > bbox_default[3] or bbox[3] < bbox_default[1])):
        #     print("IN Active Range")
        # else:
        #     print("NOT IN Active Range")
        # cv2.rectangle(img, (xmin-20 , ymin ), (xmax +20 , ymax ), (0, 0,255), 2) #active region

        if len(lmlst) != 0:
            x1, y1 = lmlst[8][1:]
            x2, y2 = lmlst[12][1:]

            fingers_upright = detector.fingersUp(img)
            # print(fingers_upright)
            if fingers_upright[1] == 1 and fingers_upright[2] == 1:
                # print("Wid,height=", c_width, c_height)
                new_x = np.interp(x1, (bbox_default[0], c_width - bbox_default[0]), (0, screen_width))
                new_y = np.interp(y1, (bbox_default[1], c_height - bbox_default[1]), (0, screen_height))

                x_after_smoothing = x_before_smoothing +(new_x - x_before_smoothing)/smooth_factor
                y_after_smoothing = y_before_smoothing + (new_y - y_before_smoothing) / smooth_factor

                cv2.circle(img, (x1, y1), 10, (123, 123, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (123, 123, 255), cv2.FILLED)

                # print("X!==", x1)
                # print(new_x, new_y)
                scale = autopy.screen.scale()
                try:
                    # autopy.mouse.smooth_move(new_x/scale, new_y/scale)
                    autopy.mouse.move(x_after_smoothing,y_after_smoothing)#(new_x, new_y)
                    x_before_smoothing, x_before_smoothing =x_after_smoothing,y_after_smoothing
                except:
                    print("OUT of Bounds")
                joints = [[8, 6, 5]]
                img, angle = detector.findAnglebetween(joints, detector.results, img)
                # print("ANGLE==", angle)
                try:
                    if angle < 160:
                        click_flag = 1
                    if click_flag == 1:
                        print("Clicking")
                        autopy.mouse.click()
                        click_flag = 0
                except:
                    print("Unable to click")

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)
