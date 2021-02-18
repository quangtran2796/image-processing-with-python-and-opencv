import numpy as np
import cv2
img_gray=cv2.imread('cat_1.jpg',0)
img_color=cv2.imread('cat_1.jpg',cv2.IMREAD_UNCHANGED)


rows, cols= img_gray.shape
img_gray_1=np.float32(img_gray)
for x in range(0,rows):
    for y in range(0,cols):
        if 0<= img_gray_1[x,y]*255<= 2:
            img_gray_1[x,y]=0
        elif 2< img_gray_1[x,y]*255<= 160:
            img_gray_1[x,y]=img_gray_1[x,y]           
        else:
            img_gray_1[x,y]=1
            
        
img_gray_2=cv2.convertScaleAbs(img_gray_1,0,255)
cv2.namedWindow('img_color', cv2. WINDOW_NORMAL)
cv2.namedWindow('img_gray_1', cv2. WINDOW_NORMAL)
cv2.imshow('img_color',img_color)
cv2.imshow('img_gray_1',img_gray_1)
cv2.imwrite('Linear_function.jpg',img_gray_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
