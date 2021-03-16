import cv2 
import numpy as np 
import math
import matplotlib.pyplot as plt
import copy

from numpy.core.records import get_remaining_size
filename = './imageset0/curve_a.jpg'

def conv_to_bw(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    z=0
    z+=min(np.ndarray.flatten(img))
    z+=max(np.ndarray.flatten(img))
    z/=2
    z=math.floor(z)
    thresh, imgbw = cv2.threshold(img, z, 255, cv2.THRESH_BINARY)
    return imgbw

def count_black_points(bwimg):
    bp=0
    for i in range(bwimg.shape[0]):
        for j in range(bwimg.shape[1]):
            if (bwimg[i][j] == 0):
                bp+=1
    return(bp)

def get_images_in_folder(foldername):
    img_list=list()
    for i in range(10000):  #remove hardlimit
        img = cv2.imread(foldername+'/'+str(i)+'.jpg')
        if  (img is None):
            break
        else:
            img_list.append(img)
    if(len(img_list)!=0):
        return img_list
    else:
        return None

def crop(filename):

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lineLength=math.floor(0.8*min(img.shape[:2]))

    lines = cv2.HoughLines(edges,1,np.pi/180, lineLength)
    crop_x=list()
    crop_y=list()
    for line in lines:
        for r,theta in line: 
            a = np.cos(theta) 
            b = np.sin(theta) 
            x0 = a*r 
            y0 = b*r  
            x1 = int(x0 + 1000*(-b)) 
            y1 = int(y0 + 1000*(a)) 
            x2 = int(x0 - 1000*(-b)) 
            y2 = int(y0 - 1000*(a)) 

            if (abs(y1-y2)<3):
                crop_y.append(y1)
            if (abs(x1-x2)<3):
                crop_x.append(x1)
            
            cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)
    crop_x=max(crop_x)
    crop_y=min(crop_y)

    img2=img[:crop_y,crop_x:]
    cv2.imwrite('./imageset0/cropped.jpg', img2)
    return img2

def getonlygraph(bwimg):

    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(bwimg,-1,kernel)
    (thresh, bwimg2) = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
    if np.unique(np.ndarray.flatten(bwimg2)).shape[0] == 1:
        return {"value": False, "img": None}
    first=None
    last=None
    
    for i in range(bwimg2.shape[0]):
        for j in range(bwimg2.shape[1]):
            if bwimg2[i][j]==0:
                if first == None or i+j<first[0]+first[1]:
                    first=(i,j)
                if last == None or i+j>last[0]+last[1]:
                    last=(i,j)  
    

    bwimg=bwimg[first[0]:last[0],first[1]:last[1]]
    if(bwimg2.size == 0):   
        return {"value": False, "img": None}
    else:
        return {"value": True, "img": bwimg}

def getlines():
    img = crop(filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    bwimages=list()
    buckets=range(0,256,10)
    x=0
    for i in buckets:
        for j in buckets:
            dummy=copy.deepcopy(img)
            mask = img[:] < i
            mask2 = img[:] > j
            mask3 = mask | mask2
            dummy[mask3]=255

            t=np.unique(np.ndarray.flatten(dummy))
            if(t.shape[0]<60 and t.shape[0]>30):
                z=0
                z+=min(np.ndarray.flatten(dummy))
                z+=max(np.ndarray.flatten(dummy))
                z/=2
                z=math.floor(z)
                thresh, dummybw = cv2.threshold(dummy, z, 255, cv2.THRESH_BINARY)
                if np.unique(np.ndarray.flatten(dummybw)).shape[0] !=1:
                    bwimages.append(dummybw)

    bwgraphs=list()
    for bwimg in bwimages:
        t = getonlygraph(bwimg)
        if(t["value"] != False):
            bwgraphs.append(t["img"])

    black_points=dict()
    x=0
    img_list=list()
    for i in range(len(bwgraphs)):
        bp=count_black_points(bwgraphs[i])
        if(bp>0):
            #cv2.imwrite(f'./imageset0/only_lines/{x}.jpg',bwgraphs[i])
            black_points[x]=bp
            x+=1
            img_list.append(bwgraphs[i])

    graph_interval =[math.floor(0.7*np.mean(list(black_points.values()))),
                    math.floor(1.6*np.mean(list(black_points.values())))] #Edit these values
    graphs=list()
    seperators=list()
    for i in black_points:
        if(black_points[i]>=graph_interval[0] and black_points[i]<=graph_interval[1]):
            graphs.append(i)
        else:
            seperators.append(i)
    sep=list()
    for i in range(len(seperators)-1):
        x=list()
        if (seperators[i] not in [item for sublist in sep for item in sublist]):
            x.append(seperators[i])
        for j in range(i+1,len(seperators)):
            if(len(sep) == 0 or seperators[j] not in [item for sublist in sep for item in sublist]):
                if(abs(seperators[j-1]-seperators[j])<=5):
                    x.append(seperators[j])
                else:
                    break
            else:
                break
        if(len(x)!=0):
            sep.append(x)
    bucket_indices=list()
    bucket_indices.append([0,sep[0][0]-1])
    for i in range(len(sep)-1):
        bucket_indices.append([max(sep[i])+1,min(sep[i+1])-1])
    bucket_indices.append([max(sep[-1])+1,graphs[-1]])
    graph_buckets=list()
    for i in bucket_indices:
        bucket=list()
        for j in graphs:
            if(j>=i[0] and j<=i[1]):
                bucket.append(j)
        graph_buckets.append(bucket)

    
    g=list()
    for bucket in graph_buckets:
        t=list()
        minsize=None
        for i in bucket:
            size=img_list[i].shape
            if(minsize is None or (size[0]<minsize[0] and size[1]<minsize[1])):
                minsize=size
        for i in bucket:
            size=img_list[i].shape
            if(size[0]<=minsize[0] or size[1]<=minsize[1]):
                t.append(i)
        g.append(t)
    graph_buckets=g   
    graphs=list()
    for bucket in graph_buckets:
        if(len(bucket)!=1):
            maxbp=None
            for i in bucket:
                if(maxbp is None or black_points[i]>maxbp):
                    maxbp=black_points[i]
            for i in bucket:
                if (black_points[i]==maxbp):
                    graphs.append(i)
                    break
        else:
            graphs.append(bucket[0])
    t=[]
    j=0
    for i in graphs:
        x=img_list[i]
        cv2.imwrite(f'./imageset0/unique_graphs/{j}.jpg',x)
        j+=1
        t.append(x)
    return t
def interpret(graphs):
    for img in graphs:
        graph_points=list()
        f=dict()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.array_equal(img[i][j],[0,0,0]):
                    graph_points.append((j,(img.shape[0]-i)))
        x,y=list(zip(*graph_points))
        y=list(set(y))
        final=dict()
        for b in y:
            bucket=list()
            for i in graph_points:
                if(i[1]==b):
                    bucket.append(i[0])
            lb=min(bucket)
            up=max(bucket)
            final[b]=(lb,up)
        print(final)
        print()

#graphs=getlines()
graphs2=get_images_in_folder('./imageset0/unique_graphs')
for i in graphs2:
    i=conv_to_bw(i)
interpret(graphs2)


