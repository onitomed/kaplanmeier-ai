import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('./imageset0/onlyline.jpg')
(thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
graph_points=list()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if np.array_equal(img[i][j],[0,0,0]):
            graph_points.append( (j,((img.shape[0]-i)/img.shape[0])) )

x,y=list(zip(*graph_points))
plt.scatter(x,y)
plt.show()