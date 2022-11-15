# -*- coding: utf-8 -*-

from multiprocessing import allow_connection_pickling
import platform
import numpy as np
import argparse
import cv2
import serial
import time
import sys
from threading import Thread
import csv
import math

ser = serial.Serial('/dev/ttyS0', 4800, timeout =1)
serial_use = 1

serial_port =  None
Temp_count = 0
Read_RX =  0

threading_Time = 5/1000.


def TX_data(ser, one_byte):  # one_byte= 0~255
    ser.write(chr(int(one_byte)))          #python2.7
    #ser.write(serial.to_bytes([one_byte]))  #python3
#-----------------------------------------------
def RX_data(serial):
    global Temp_count
    try:
        if serial.inWaiting() > 0:
            result = serial.read(1)
            RX = ord(result)
            return RX
        else:
            return 0
    except:
        Temp_count = Temp_count  + 1
        print("Serial Not Open " + str(Temp_count))
        return 0
        pass
        
            
def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()
            

def nothing(x):
	# any operation
	pass

def allow_recog():
    for i in range(100):
        _,frame = cap.read()

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L-H","Trackbars")
        l_s = cv2.getTrackbarPos("L-S","Trackbars")
        l_v = cv2.getTrackbarPos("L-V","Trackbars")
        u_h = cv2.getTrackbarPos("U-H","Trackbars")
        u_s = cv2.getTrackbarPos("U-S","Trackbars")
        u_v = cv2.getTrackbarPos("U-V","Trackbars")

    #위에 트랙바 쓸경우
        lower_black = np.array([l_h, l_s, l_v])
        upper_black = np.array([u_h, u_s, u_v])

    # 트랙바 안쓸 경우
    #lower_black = np.array([0, 0, 0])
    #upper_black = np.array([180, 235, 150])
    
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_black, upper_black)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel)

        contours,_ = cv2.findContours(mask , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        result_allow = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True),True)

            

        #상황애 따라 조절
            if area > 100 :
                
                points = []
                if len(approx)==7:

                    for i in range(7):
                        points.append([approx.ravel()[2*i], approx.ravel()[2*i+1]])

                        points.sort()

                    minimum = points[1][0] - points[0][0]
                    maximum = points[6][0] - points[5][0]

                    cv2.drawContours(frame,[approx],0,(0,255,0),5)
                
                    if maximum < minimum :
                        cv2.putText(frame, "left", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2 ,cv2.LINE_AA)
                        result_allow = 1

                    elif maximum > minimum :
                        cv2.putText(frame, "right", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2 ,cv2.LINE_AA)
                        result_allow = 2

                    else:
                        result_allow = 100
        print(result_allow)  
        a.insert(i, result_allow)               
        #time.sleep(1)

        cv2.imshow("Frame",frame)
        cv2.imshow("MASK",mask)
        key = cv2.waitKey(1) 
        if key ==27:
            break
    add = sum(a)
    addL1 = (add / 100)
    addL2 = (add % 100)
    
    print(add)
    print(addL2)
    if (addL2 > 80 and addL2 < 110):
        TX_data(serial_port, 5)
    elif (addL2 < 220 and addL2 > 180):
        TX_data(serial_port, 6)
    else:
        print('None')
        
hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

def line_track():
    while(True):
        
        ret,img_ori = cap.read()
        img_color = cv2.resize(img_ori, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    #img_color = cv.resize(img_ori, dsize=(640, 480), interpolation=cv.INTER_AREA)
    #height, width = img_color.shape[:2]
    #img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)

        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([19,30,30])          # 파랑색 범위
        upper_blue = np.array([31,255,255])

        lower_green = np.array([9, 30, 30])        # 초록색 범위
        upper_green = np.array([21, 255, 255])

        lower_red = np.array([9, 30, 30])        # 빨강색 범위
        upper_red = np.array([21, 255, 255])

    # Threshold the HSV image to get only blue colors
        img_mask1 = cv2.inRange(img_hsv, lower_blue, upper_blue)     
        img_mask2 = cv2.inRange(img_hsv, lower_green, upper_green) 
        img_mask3 = cv2.inRange(img_hsv, lower_red, upper_red)
        img_mask = img_mask1 | img_mask2 | img_mask3

    # Bitwise-AND mask and original image
        img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask) 

        cdst = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(img_result, 50, 200, None, 3)
    
        _, binary = cv2.threshold(cdst, 0, 255, cv2.THRESH_BINARY)
    
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

        cv2.line(img_color, (320, 0),(320, 480),(125,0,0),2)

        contours1, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
        for cnt in contours1:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True),True)
        #상황애 따라 조절
            if area > 1500 :

                points = []
                a = []
                b = []
            
                if len(approx)==4:

                    for i in range(4):
                        points.append([approx.flatten()[2*i], approx.flatten()[2*i+1]])

                    #points.sort()

                        cv2.drawContours(img_color,[approx],0,(0,255,0),2)
                    
                        if points[i][1] > 300:
                            a.insert(i, points[i])
                            #print(points[i])
                        elif points[i][1] < 200:
                            b.insert(i, points[i])
                            #print(points[i])
                    up_x = int((a[0][0] + a[1][0]) / 2)
                    up_y = int((a[0][1] + a[1][1]) / 2)
                
                    down_x = int((b[0][0] + b[1][0]) / 2)
                    down_y = int((b[0][1] + b[1][1]) / 2)
                
                    center_x = int((up_x + down_x)/2)
                    center_y = int((up_y + down_y)/2)
                #print(a)
                    cv2.line(img_color, (up_x, up_y),(down_x, down_y),(125,0,0),2)
                    cv2.line(img_color, (up_x, up_y),(up_x, down_y),(125,0,0),2)
                    cv2.circle(img_color,(center_x, center_y),5,(125,0,0),-1)
                
                    angle = np.arctan((up_x - down_x) / (up_y - down_y)) 
                    distance = center_x -320
                #print(angle)
                    print("d : ",distance, "  angle : ", angle)  
                    
                    if (distance > 100):
                        TX_data(serial_port, 20)
                    elif (distance < -100):
                        TX_data(serial_port, 15)
                    #elif (angle < -0.9):
                        #TX_data(serial_port, 9)
                    #elif (angle > 0.9):
                        #TX_data(serial_port, 7)
                    else:
                        TX_data(serial_port, 2)
            
        cv2.imshow('img_color', img_color)
        cv2.imshow('Binary', opening)
        
        
        
        if (cv2.waitKey(1)) & 0xFF == 27:
            break        
                    
BPS =  4800  # 4800,9600,14400, 19200,28800, 57600, 115200
serial_use = 1

if serial_use <> 0:
        BPS =  4800  # 4800,9600,14400, 19200,28800, 57600, 115200
        #---------local Serial Port : ttyS0 --------
        #---------USB Serial Port : ttyAMA0 --------
        serial_port = serial.Serial('/dev/ttyS0', BPS, timeout=0.01)
        serial_port.flush() # serial cls
        time.sleep(0.5)
    
        serial_t = Thread(target=RX_data, args=(serial_port,))
        serial_t.daemon = True
        serial_t.start()
        
    # First -> Start Code Send 
TX_data(serial_port, 1)
TX_data(serial_port, 1)
TX_data(serial_port, 1)
    
old_time = clock()

View_select = 0
msg_one_view = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H","Trackbars",0,180,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cv2.createTrackbar("U-H","Trackbars",180,180,nothing)
cv2.createTrackbar("U-S","Trackbars",236,255,nothing)
cv2.createTrackbar("U-V","Trackbars",100,255,nothing)

font = cv2.FONT_HERSHEY_COMPLEX

line_track()

cap.release()
cv2.destroyAllWindows()	
