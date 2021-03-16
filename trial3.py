import cv2 
import numpy as np 

img2 = cv2.imread('./imageset0/single_line.jpg')
(thresh, img) = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
#cv2.imwrite('./imageset0/binary.jpg',bw)

first=None
last=None
print(len(img))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if np.array_equal(img[i][j],[0,0,0]):
            if first == None or i+j<first[0]+first[1]:
                first=(i,j)
            if last == None or i+j>last[0]+last[1]:
                last=(i,j)  
print(first,last)
