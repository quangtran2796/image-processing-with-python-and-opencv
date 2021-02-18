import numpy as np
import cv2
img_gray=cv2.imread('cat_1.jpg',0)
img_color=cv2.imread('cat_1.jpg',cv2.IMREAD_UNCHANGED)


img_gray_1=np.float32(img_gray)
img_gray_1=cv2.pow(img_gray_1,25)
img_gray_2=cv2.convertScaleAbs(img_gray_1,0,255)


cv2.namedWindow('img_color', cv2. WINDOW_NORMAL)
cv2.namedWindow('Power_function', cv2. WINDOW_NORMAL)
cv2.imshow('img_color',img_color)
cv2.imshow('Power_function',img_gray_1)
cv2.imwrite('Power_function.jpg',img_gray_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
