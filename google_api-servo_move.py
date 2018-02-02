""" Servo moves if camera detects a face: using opencv"""
# credit to https://www.superdatascience.com/opencv-face-detection/ for most of the code structure

#import required libraries 
#import OpenCV library
import cv2
#import matplotlib library
import matplotlib.pyplot as plt
#importing time library for servo movement
import time 
#importing GPIO to move servos via gpio pins
import GPIO


#load test iamge
test1 = cv2.imread('data/test1.jpg')
#convert the test image to gray image as opencv face detector expects gray images 
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

#load test iamge
test1 = cv2.imread('data/test1.jpg')
#convert the test image to gray image as opencv face detector expects gray images 
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

#if you have matplotlib installed then  
plt.imshow(gray_img, cmap='gray')  

#load cascade classifier training file for haarcascade WITH CORRECT FILE PATH for .xml file
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')



