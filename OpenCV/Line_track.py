# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

def nothing(x):
    	# any operation
	pass

def mouse_callback(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3, threshold

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        print(img_color[y, x])
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        threshold = cv.getTrackbarPos('threshold', 'img_result')
        
        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv[0] < 10:
            print("case1")
            lower_blue1 = np.array([hsv[0]-10+180, threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], threshold, threshold])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([hsv[0]+10, 255, 255])
            lower_blue2 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0])
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)
        


cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)

cv.namedWindow('img_result')
cv.createTrackbar('threshold', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'img_result', 30)

cap = cv.VideoCapture(0)
    
    
while(True):
     ret,img_color = cap.read()
     height, width = img_color.shape[:2]
     img_color = cv.resize(img_color, (640, 360), interpolation=cv.INTER_AREA)
     dst = cv.Canny(img_color, 50, 200, None, 3)
     cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    
     linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

     if linesP is not None:   
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

     # 원본 영상을 HSV 영상으로 변환합니다.
     img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

     # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
     img_mask1 = cv.inRange(img_hsv, lower_blue1, upper_blue1)
     img_mask2 = cv.inRange(img_hsv, lower_blue2, upper_blue2)
     img_mask3 = cv.inRange(img_hsv, lower_blue3, upper_blue3)
     img_mask = img_mask1 | img_mask2 | img_mask3

     kernel = np.ones((11,11), np.uint8)
     img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
     img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)

     # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
     img_result = cv.bitwise_and(cdst, cdst, mask=img_mask)


     numOfLabels, img_label, stats, centroids = cv.connectedComponentsWithStats(img_mask)

     for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x,y,width,height,area = stats[idx]
        centerX,centerY = int(centroid[0]), int(centroid[1])
        print(centerX, centerY)

        
     frame_add = cv.addWeighted(img_color, 0.9, img_result, 0.3, 0)
     
     cv.imshow('img_color', img_color)
     


    # ESC 키누르면 종료
     if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
