import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = R'C:/Program Files/Tesseract-OCR'
config = ('-l kor+eng --oem 3 --psm 11')

frame = None

inputmode = False

cap = cv.VideoCapture(1)

def Text_recog():
    if cap.isOpened():
        while True:
            ret, img_color = cap.read()
            cv.imshow('cam', img_color)
            k = cv.waitKey(1)
            if k == 27:
                break

# 'i' 버튼 클릭시 객체선택 활성화
            if k == ord('i'):
                print("The text is")
                inputmode = True
# 기존에 선택되었던 객체영역에 대한 값 초기화
                frame = img_color.copy()
                img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                blur = cv.GaussianBlur(img_gray,(3,3),0)
                canny = cv.Canny(blur,100,200)
                contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                img_contour = cv.drawContours(frame, contours, -1, (0, 0, 0), 1)

                contour_pos = []

# 면적이 500 이상인 컨투어 영역만 추출
                for pos in range(len(contours)):
                    area = cv.contourArea(contours[pos])
                    if area > 500:
                        contour_pos.append(pos)

                contours_xy = np.array(contours)
                contours_xy.shape

                

                
            
                cv.imwrite('C:/Sajin/thresh1.png', img_contour)

                while inputmode:
                    cv.imshow('frame', img_contour)
                    im = cv.imread('C:/Sajin/thresh1.png')
                
                    text = (pytesseract.image_to_string(im,config="--psm 10"))
                    #ccc
                    print(text)
                    cv.imshow('ori',im)
                
                    k = cv.waitKey(0)
                    if k == 27:
                        break



cap.release()
cv.destroyAllWindows()
