# -*- coding: utf-8 -*-
#마우스 위치의 색을 추출해 내는 코드
#프레임 속도가 안나오는 이유 - 영상을 hsv로 변환 하기 떄문 이라고 생각
######################################################################
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
        
def RX_Receiving(ser):
    global receiving_exit,threading_Time

    global X_255_point
    global Y_255_point
    global X_Size
    global Y_Size
    global Area, Angle


    receiving_exit = 1
    while True:
        if receiving_exit == 0:
            break
        time.sleep(threading_Time)
        
        while ser.inWaiting() > 0:
            result = ser.read(1)
            RX = ord(result)
            print ("RX=" + str(RX))
            
def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()
            

def nothing(x):
	# any operation
	pass
#####마우스 이벤트 함수
def mouse_callback(event, x, y, flags, param):
    global hsv, lower_1, upper_1, lower_2, upper_2, lower_3, upper_3

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환 (x, y 값으로 저장)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(frame[y, x])
        color = frame[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv2.cvtColor(one_pixel, cv2.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 픽셀값의 범위를 정함
        if hsv[0] < 10:
            print("case1")
            lower_1 = np.array([hsv[0]-10+180, 30, 30]) # 색상만 조절
            upper_1 = np.array([180, 255, 255])
            lower_2 = np.array([0, 30, 30])
            upper_2 = np.array([hsv[0], 255, 255])
            lower_3 = np.array([hsv[0], 30, 30])
            upper_3 = np.array([hsv[0]+10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_1 = np.array([hsv[0], 30, 30])
            upper_1 = np.array([180, 255, 255])
            lower_2 = np.array([0, 30, 30])
            upper_2 = np.array([hsv[0]+10-180, 255, 255])
            lower_3 = np.array([hsv[0]-10, 30, 30])
            upper_3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_1 = np.array([hsv[0], 30, 30])
            upper_1 = np.array([hsv[0]+10, 255, 255])
            lower_2 = np.array([hsv[0]-10, 30, 30])
            upper_2 = np.array([hsv[0], 255, 255])
            lower_3 = np.array([hsv[0]-10, 30, 30])
            upper_3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(f"h:{hsv[0]}\nv:{hsv[1]}\ns:{hsv[2]}")
        print("@1", lower_1, "~", upper_1)
        print("@2", lower_2, "~", upper_2)
        print("@3", lower_3, "~", upper_3)

BPS =  4800  # 4800,9600,14400, 19200,28800, 57600, 115200
serial_use = 1 

if serial_use <> 0:
        BPS =  4800  # 4800,9600,14400, 19200,28800, 57600, 115200
        #---------local Serial Port : ttyS0 --------
        #---------USB Serial Port : ttyAMA0 --------
        serial_port = serial.Serial('/dev/ttyS0', BPS, timeout=0.01)
        serial_port.flush() # serial cls
        time.sleep(0.5)
    
        serial_t = Thread(target=RX_Receiving, args=(serial_port,))
        serial_t.daemon = True
        serial_t.start()
        
    # First -> Start Code Send 
TX_data(serial_port, 1)
TX_data(serial_port, 1)
TX_data(serial_port, 1)
##################################로봇 영상 기본 설정 640,480    
old_time = clock()

View_select = 0
msg_one_view = 0

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_callback)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#####################################
while(True):
    _,frame =cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # imh_hsv=frame.copy()
    img_mask1 = cv2.inRange(frame_hsv, lower_1, upper_1)
    img_mask2 = cv2.inRange(frame_hsv, lower_2, upper_2)
    img_mask3 = cv2.inRange(frame_hsv, lower_3, upper_3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    img_result = cv2.bitwise_and(frame, frame, mask=img_mask)


    cv2.imshow('frame', frame)
    cv2.imshow('img_mask', img_mask)
    cv2.imshow('img_result', img_result)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()