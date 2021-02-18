import numpy as np
import cv2
im_gray = cv2.imread('C:\Users\quang\Desktop\Lenna.png',0)
im_bw = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)[1]
cv2.namedWindow('image',cv2. WINDOW_NORMAL)
cv2.imshow('image',im_bw)
while(True):
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        cv2.destroyAllWindows()
        break;
    elif k == ord('s'):
        cv2.imwrite('binarypicture.jpg',im_bw)
        cv2.destroyAllWindows()
        break; 
