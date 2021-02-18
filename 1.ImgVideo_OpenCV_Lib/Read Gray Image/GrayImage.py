import numpy as np
import cv2
img = cv2.imread('D:\Lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('D:\homework1.png',img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

