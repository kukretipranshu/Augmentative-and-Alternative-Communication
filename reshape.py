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
path = r'D:\majorproject\Augmentative-and-Alternative-Communication\dataset\test_set' # Source Folder
dstpath = r'D:\majorproject\Augmentative-and-Alternative-Communication\mydata' # Destination Folder
dstpath1 = r'D:\majorproject\Augmentative-and-Alternative-Communication\mydata' # Destination Folder

os.makedirs(dstpath,exist_ok=True)
# os.makedirs(os.path.join(dstpath,'train_set'),exist_ok=True)
# dstpath = os.path.join(dstpath,'train_set')
os.makedirs(os.path.join(dstpath,'test_set'),exist_ok=True)
dstpath = os.path.join(dstpath,'test_set')

files = os.listdir(path)

# for file in files:
#     curr_file = file
#     os.makedirs(os.path.join(dstpath,file),exist_ok=True)
#     file = os.listdir(os.path.join(path,curr_file))
#     for image in file:
#         print(image)
#         # print(os.path.join(path,curr_file,image))
#         img = cv2.imread(os.path.join(path,curr_file,image))
#         gray = cv2.resize(img, (300,300))
#         gray = gray[:,:,:1]
#         # print(gray.shape)
#         cv2.imwrite(os.path.join(dstpath,curr_file,image),gray)


img = cv2.imread(os.path.join(dstpath,'A','adb57198-aae6-11ed-a1e4-085bd6f0acd4.jpg'))
print(img.shape)
img = cv2.imread(os.path.join(dstpath1,'train_set','A','0.jpg'))
print(img.shape)