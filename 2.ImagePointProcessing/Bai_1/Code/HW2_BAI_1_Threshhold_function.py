import numpy as np
import cv2
img_gray=cv2.imread('cat_1.jpg',0)
img_color=cv2.imread('cat_1.jpg',cv2.IMREAD_UNCHANGED)


img_gray_1=cv2.threshold(img_gray,225,255,cv2.THRESH_BINARY)[1]

        
cv2.namedWindow('img_color', cv2. WINDOW_NORMAL)
cv2.namedWindow('img_gray_1', cv2. WINDOW_NORMAL)
cv2.imshow('img_color',img_color)
cv2.imshow('img_gray_1',img_gray_1)
cv2.imwrite('ThreshHold_Function.jpg',img_gray_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
