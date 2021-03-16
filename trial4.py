import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img2 = cv2.imread('./imageset0/single_line.jpg')
(thresh, img) = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(img,100,200)
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
(thresh, dst) = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
img = dst
first=None
last=None
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if np.array_equal(img[i][j],[0,0,0]):
            if first == None or i+j<first[0]+first[1]:
                first=(i,j)
            if last == None or i+j>last[0]+last[1]:
                last=(i,j)  
img2=img2[first[0]:last[0],first[1]:last[1]]
(thresh, img) = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('./imageset0/onlyline.jpg',img)
