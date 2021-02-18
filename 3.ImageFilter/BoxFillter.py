import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Image2.jpg')
blur = cv2.blur(img,(7, 7))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Nornalized Box Filter')
plt.xticks([]), plt.yticks([])
plt.show()
