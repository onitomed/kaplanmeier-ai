import cv2 
import numpy as np 

img = cv2.imread('./imageset0/cropped.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("./imageset0/grayimg.jpg",gray)
img = cv2.imread('./imageset0/grayimg.jpg')


mask=img[:,:,1] > 50
img[mask]=255

cv2.imwrite("./imageset0/single_line.jpg",img)
