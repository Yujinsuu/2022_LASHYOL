# -*- coding: utf-8 -*-

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
    if serial.inWaiting() > 0:
        result = serial.read()
        if result == "":
            print("=====404 NOT FOUND=====")
        RX = ord(result)
        return RX
    else:
        return 0
    
            
def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()
            

def nothing(x):
	# any operation
	pass


def line_trace(img_color):
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    #색상영역 지정
    lower_yellow = np.array([20,50,50])
    upper_yellow = np.array([35,255,255])

    # Threshold the HSV image to get only blue colors
    img_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

    return img_result


            
def distance(x0,y0,x1,y1):
    dx = x0 - x1
    dy = y0 - y1
    dist = math.sqrt((dx*dx)+(dy*dy))
    return dist

            
def find_area(x0,y0,x1,y1,h0,w0,h1,h2,w1,w2):
    if (x0 in range(w1,w2)) and (y0 in range(0,h1)):
        area0=1
    elif (x0 in range(0,w1)) and (y0 in range(h1,h2)):
        area0=2
    elif (x0 in range(w1,w2)) and (y0 in range(h2,h0)):
        area0=3
    elif (x0 in range(w2,w0)) and (y0 in range(h1,h2)):
        area0=4
    else:
        area0=0
        
    if (x1 in range(w1,w2)) and (y1 in range(0,h1)):
        area1=1
    elif (x1 in range(0,w1)) and (y1 in range(h1,h2)):
        area1=2
    elif (x1 in range(w1,w2)) and (y1 in range(h2,h0)):
        area1=3
    elif (x1 in range(w2,w0)) and (y1 in range(h1,h2)):
        area1=4
    else:
        area1=0
        
    return area0,area1

def identify(a0,a1):
    if a0 != a1:
        if a0 * a1 == 0:
            sign0 = {1:1, 2:3, 3:2, 4:4}.get(a0 + a1, None)
            # 1:up, 2:down, 3:left, 4:right
            return [0,sign0]
        else:
            if (a0 + a1) % 2 == 0:
                if (a0 * a1) % 2 == 0:
                    sign1 = [3,4]
                else:
                    sign1 = [1,2]
                return [1,sign1]

def corner_check(msg):
    count1=0
    count2=0
    count3=0
    count4=0
    for i in range(100):
        
        ret,img_ori = cap.read()
        
        img_color = cv2.resize(img_ori, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        
        img = line_trace(img_color)
        height, width = img.shape[:2] # 이미지 높이, 너비

        cdst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(img, 50, 200, None, 3)
    
        _, binary = cv2.threshold(cdst, 0, 255, cv2.THRESH_BINARY)
    
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

        contour_C, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.drawContours(dst,contour_C,-1,(255,0,0),2)

        lines = cv2.HoughLinesP(contour_img,1,1*np.pi / 180, 60, np.array([]),minLineLength = 10, maxLineGap = 100)
        hough_img = np.zeros((img_ori.shape[0], img_ori.shape[1], 3), dtype=np.uint8)
    
        get_sign=[]
        sign=[]
        #검출 라인 분석
        if lines is not None:
            point=[]
            h1,h2,w1,w2 = int(height/2 - 100), int(height/2 + 100), int(width/2 - 100), int(width/2 + 100)
            for line in lines:
                x0,y0,x1,y1 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])
                if distance(x0,y0,x1,y1) >= 100:
                    point.append(x0)
                    point.append(y0)
                    point.append(x1)
                    point.append(y1)
                    cv2.line(hough_img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    if (x0 > w1 and x0 < w2) or (y0 > h1 and y0 < h2):
                        if (x1 > w1 and x1 < w2) or (y1 > h1 and y1 < h2):
                            area0,area1 = find_area(x0,y0,x1,y1,height,width,h1,h2,w1,w2)
                            try:
                                ID = identify(area0,area1)
                                if ID[0] == 0:
                                    get_sign.append(ID[1])
                                elif ID[0] == 1:
                                    get_sign.extend(ID[1])
                            except:
                                pass
                            
            sign = list(set(get_sign))
        if 1 in sign: count1 += 1
        if 2 in sign: count2 += 1
        if 3 in sign: count3 += 1
        if 4 in sign: count4 += 1
    
    sign=[]
    if count1/100 > 0.7: sign.apeend(1)
    if count2/100 > 0.7: sign.apeend(2)
    if count3/100 > 0.7: sign.apeend(3)
    if count4/100 > 0.7: sign.apeend(4)
    
    if msg == 100 or msg == 101:
        if sign == [2,3,4]: return 114
        else: return 0


def line_track():
    dist=0
    for i in range(100):
        ret,img_ori = cap.read()
        
        img_color = cv2.resize(img_ori, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        
        img_result = line_trace(img_color)

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
                    dist += distance
    if (dist < -100): return 112
    elif (dist > 100): return 113
    else: return 111
            
         
         


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

while True:
    global rx
    serial_port.flush()
    rx = RX_data(serial_port)
    print('rx = %d',rx)
    # start
    if rx == 1: TX_data(serial_port,110)
    # go straight
    elif rx in [100,101,112,113,120]: 
        mode = corner_check(rx)
        if mode == 0: mode = line_track()
        TX_data(serial_port,mode)
        print('tx = %d',mode)
    # Alphabet
    if (cv2.waitKey(1)) & 0xFF == 27:
            break 


cap.release()
cv2.destroyAllWindows()	
