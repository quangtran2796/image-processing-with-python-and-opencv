import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('Image2.jpg')

median = cv2.medianBlur(img,7)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Median Filter')
plt.xticks([]), plt.yticks([])
plt.show()
