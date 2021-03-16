import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import cv2

pic = imageio.imread("./imageset0/curve_a.jpg")
gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
graypic = gray(pic)

x_base = math.floor(0.8*pic.shape[0])
y_base = math.floor(0.8*pic.shape[1])
print(x_base,y_base)
#finding axes, assuming single pixel wide axes

for i in range(len(pic)):
    for j in range(i):
        if(not (np.array_equal(pic[i][j],[255,255,255]) or np.array_equal(pic[i][j],[0,0,0]))):
            print(i,j)
        break

# img = Image.open("./imageset0/curve_a.jpg").convert('L')
# img.save('./imageset0/curve_a_gray.jpg')
# print(img.tostring())


# plt.figure(figsize = (5,5))
# plt.imshow(img)
#plt.show()

# for i in pic:
#     for j in i:
#         if(j.all() == [255,255,255]):
#             j = [0,0,0]
