"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np
from matplotlib import pyplot as plt


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        # self.joints= [[8,5,0],[20,17,0]] #(tip,joint,andwrist) for index, middle=[12,9,0],,pinky
        self.joints = [[8,6,0]] #,[20,18,0]]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for num_hands_index, handLms in enumerate(self.results.multi_hand_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(255, 234, 91), thickness=2,
                                                                       circle_radius=4),
                                               self.mpDraw.DrawingSpec(color=(134, 22, 126), thickness=2,
                                                                       circle_radius=2), )
                if self.LeftRight(num_hands_index, handLms, self.results, img):
                    text, coords = self.LeftRight(num_hands_index, handLms, self.results, img)
                    cv2.putText(img, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                img= self.findAnglebetween(self.joints,self.results,img)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # print("MyHand==",myHand)
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                depth_z = int(lm.z * (w * h))
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                # if draw:
                    # cv2.circle(img, (cx, cy), 5, (255, 0, 255), int(cv2.FILLED * depth_z / 10000))
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def LeftRight(self, num_hands_index, hand_landmarks, results, cam_img):
        # results= self.results
        output = None
        for index, classification in enumerate(results.multi_handedness):
            # if the classsification index is same as the hand indx in the scene
            if classification.classification[0].index == num_hands_index:
                label = classification.classification[0].label
                score = classification.classification[0].score
                text = '{} {}'.format(label, round(score, 2))

                # Exctracting coordinates of hands wrist
                coordinates = tuple(np.multiply(
                    np.array((hand_landmarks.landmark[self.mpHands.HandLandmark.WRIST].x,
                              hand_landmarks.landmark[self.mpHands.HandLandmark.WRIST].y)),
                    [640, 480]).astype(
                    int))  # cam_img.get(cv2.CV_CAP_PROP_FRAME_WIDTH), cam_img.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)
                output = text, coordinates
        return output

        # joints is a list of lists with landmarks of different fingers between which you want to calculate the angles.

    def findAnglebetween(self, joints, results, image):
        for hand in results.multi_hand_landmarks:
            # looping through joints
            for joint in joints:
                # find angle between these 3 coordinates at b
                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])

                rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                deg = np.abs(rad * 180.0 / np.pi)
                if deg > 180.0:
                    deg = 360 - deg
                cv2.putText(image, str(round(deg, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        return image


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        # print( "cam Dimensions ==",cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        # if len(lmList) != 0:
            # print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # cv2.putText(img, 'fps='+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
