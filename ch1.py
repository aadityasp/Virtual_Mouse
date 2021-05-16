# https://www.youtube.com/watch?v=WQeoO7MI0Bs       #
# -------------------------------------------------------
import cv2
import numpy as np
print("package imported")
#---------------------------------------------
# video capture and image show
# img = cv2.imread("./Resources/gem2_original.jpg")
# cv2.imshow("Output",img)
# capture_object = cv2.VideoCapture("./Resources/door_prob.mp4")
# webcam_object = cv2.VideoCapture(0)
# webcam_object.set(3,640) #width
# webcam_object.set(4,480) #height
# webcam_object.set(10,100) #brightness id =10
# while True:
#     success , img = webcam_object.read()
#     cv2.imshow("Video window", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# ------------------------------------------------------------------
# playing with edges, and blurring
# img = cv2.imread("./Resources/gem2_original.jpg")
# img = cv2.imread("./Resources/hand.jpg")
# kernel = np.ones((5,5),np.uint8)
# img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_gray,(7,7),0)
# img_canny = cv2.Canny(img,100,200)
# img_dilation = cv2.dilate(img_canny,kernel, iterations=1)
# img_eroded = cv2.erode(img_dilation, kernel, iterations=1)
#
# cv2.imshow("Gray scale image ", img_gray)
# cv2.imshow("Blur image ", img_blur)
# cv2.imshow("canny edges image ", img_canny)
# cv2.imshow(" Dilated image ", img_dilation)
# cv2.imshow(" Eroded image ", img_eroded)
# cv2.waitKey(0000)
# --------------------------------------------------------------------#
#resizeing and cropping
# img = cv2.imread("./Resources/gem2_original.jpg")
# print(img.shape)
# img_resize= cv2.resize(img,(500,700))
# print(img_resize.shape)
# img_cropped= img[0:500, 0:400]
# cv2.imshow('Image',img)
# cv2.imshow('Resized Image',img_resize)
# cv2.imshow('Cropped Image',img_cropped)
# cv2.waitKey(0)
# -----------------------------------------------------------------------#
# Lines and shapes
# img = np.zeros((512,512,3),np.uint8)
# # img[:]= 255,255,0 #BGR
# # print(img)
# cv2.line(img, (0,0), (img.shape[1],img.shape[0]),(0,255,0))
# # cv2.rectangle(img, (0,0), (250,350),(0,0,255),cv2.FILLED)
# cv2.rectangle(img, (0,0), (250,350),(0,0,255),20)
# cv2.circle(img, (400,50), 30 , (255,0,0),5)
# cv2.putText(img,"Testing Text",(300,150),cv2.FONT_ITALIC, 1, (0,150,0),2)
# cv2.imshow('Image',img)
# cv2.waitKey(0)
# ----------------------------------------------------------------------------#
#Warping images, try again
# img = cv2.imread("./Resources/hand.jpg")
# img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Image", img)
# # (200,160),(350,160),(250,337),(339,327)
# width, height =  200,200
# points= np.float32([[160,200],[160,450],[337,250],[327,300]])
# points2= np.float32([[0,0],[0,height],[width,0],[width,height]])
# matrix = cv2.getPerspectiveTransform(points,points2)
# img_output = cv2.warpPerspective(img_gray, matrix,(width,height))
# cv2.imshow("Image", img)
# cv2.imshow("Warp output ", img_output)
# cv2.waitKey(0)
# ---------------------------------------------------------------------------------#
# joining images, Watch this youtube video for custom  stacking function
img = cv2.imread("./Resources/hand.jpg")
img_horizontal = np.hstack((img,img))
img_vertical = np.vstack((img,img))
cv2.imshow("Horizontal stack  ", img_horizontal)
cv2.imshow("Vertical stack  ", img_vertical)

cv2.waitKey(0)

------------------------------------------------------------------------------------#