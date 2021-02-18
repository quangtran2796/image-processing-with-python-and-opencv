import numpy as np
import cv2
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('BinaryVideo.avi',-1,20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_bw = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)[1]
        out.write(im_bw)
    cv2.imshow('im_bw',im_bw)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

