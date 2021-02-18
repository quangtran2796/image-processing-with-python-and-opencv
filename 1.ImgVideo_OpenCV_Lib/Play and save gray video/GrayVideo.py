import numpy as np
import cv2
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('GrayVideo.avi',-1,20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(im_gray)
    cv2.imshow('im_gray',im_gray)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
