import numpy as np
import cv2
img_gray=cv2.imread('cat_1.jpg',0)
img_color=cv2.imread('cat_1.jpg',cv2.IMREAD_UNCHANGED)

img_gray_1=np.float32(img_gray)
img_gray_1=0.125*cv2.log(img_gray_1+1)
img_gray_2=cv2.convertScaleAbs(img_gray_1,0,255)


cv2.namedWindow('img_color', cv2. WINDOW_NORMAL)
cv2.namedWindow('Log_function', cv2. WINDOW_NORMAL)
cv2.imshow('img_color',img_color)
cv2.imshow('Log_function',img_gray_1)
cv2.imwrite('Log_function.jpg',img_gray_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
