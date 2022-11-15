# -*- coding: utf-8 -*-
#색(파란색)인식하여 이민 여부 판단 기본적 색인식 코드
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

def checking():
    check=0
    while check != 38:
        check=RX_data(serial)

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
count=0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
a1=640//3
a2=a1+640//3
b1=480//3
b2= b1+480//3
##################################### 색 설정
lower= np.array([175,30,30])
upper= np.array([180,255,255])
kernel = np.ones((5, 5), np.uint8)


while True:
    rx=RX_data(serial_port)
    print(rx)
    print(type(rx))
    if rx==1:
        print("good")
    else:
        print("fail")
    
