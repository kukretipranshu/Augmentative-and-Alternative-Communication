from cvzone.HandTrackingModule import HandDetector
import string
import numpy as np
import os
import cv2
# import imutils
import mediapipe as mp
import uuid

# Setting up hand points variable and landmark drawing variable

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

minValue = 70
path = r'D:\majorproject\Augmentative-and-Alternative-Communication\ISL_Folder' # Source Folder
dstpath = r'D:\majorproject\Augmentative-and-Alternative-Communication\todata' # Destination Folder



# try:
# os.makedirs(dstpath)
# except:
#     print ("Directory already exist, images will be written in asme folder")

# Folder won't used
files = os.listdir(path)

for file in files:
    curr_file = file
    if not os.path.exists(os.path.join(dstpath,file)):
        os.makedirs(os.path.join(dstpath,file))
    file = os.listdir(os.path.join(path,curr_file))
    for image in file:
        print(image)
        print(os.path.join(path,curr_file,image))
        img = cv2.imread(os.path.join(path,curr_file,image))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # blur = cv2.GaussianBlur(gray,(5,5),2)

        # th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        # ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        results = hands.process(gray)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = gray.shape

                    # Finding the coordinates of each landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # Printing each landmark ID and coordinates
                    # on the terminal
                    print(id, cx, cy)

                    # Drawing the landmark connections
                    mpDraw.draw_landmarks(gray, handLms, mpHands.HAND_CONNECTIONS)
            # return gray
        gray = cv2.Canny(gray,80,150)
        # gray = cv2.resize(gray, (300,300))
        cv2.imwrite(os.path.join(dstpath,curr_file,image),gray)