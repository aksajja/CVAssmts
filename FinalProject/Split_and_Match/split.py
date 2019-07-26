#Date: 28-02-2019
#author : 2018701022
#paperlink: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjtqMnxpODgAhVFq48KHeiHCnMQFjABegQIBRAB&url=https%3A%2F%2Fwww.researchgate.net%2Fpost%2FHow_to_calculate_variance_and_standard_deviation_of_pixels_of_an_image_3_x_3_in_matlab&usg=AOvVaw3HE9_8Pxyk5Zd5zu5M_7Qq

import cv2
import json
import numpy as np
import collections
import math
from itertools import islice
from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
from skimage.util import view_as_windows
    
def quadtree1(start, end):
    print(start, end)
    x = int(abs(start[0] - end[0])/2)
    y = int(abs(start[1] - end[1])/2)
    print(x,y)
    quad1 = [[start[0], start[1]], [start[0]+x, start[1]+y]]
    quad2 = [[start[0], start[1]+y], [start[0]+x, start[1]+y+y]]
    quad3 = [[start[0]+x, start[1]], [start[0]+x+x, start[1]+y]]
    quad4 = [[start[0]+x, start[1]+y], [start[0]+x+x, start[1]+y+y]]
    '''else:
        x = int(abs(start[0] - end[0])/2)
        y = round(abs(start[1] - end[1])/2)
        print(x,y)
        quad1 = [[start[0], start[1]], [start[0]+x, start[1]+x]]
        quad2 = [[start[0], start[1]+x], [start[0]+y, start[1]+x+y]]
        quad3 = [[start[0]+x, start[1]], [start[0]+x+y, start[1]+y]]
        quad4 = [[start[0]+y, start[1]+y], [start[0]+x+y, start[1]+x+y]] '''       
    quad = [quad1, quad2, quad3, quad4]
    print(quad)
    return(quad)
    
def BGR2YUV(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    
    y, u, v = cv2.split(img)
    return y
    
def min_distance(patch, size, v):
    p = patch.mean()
    min_d = 999999999
    dist = {}
    x1, x2 = 0,0
    y1, y2 = size[0], size[1]
    c = 0
    for pat1 in (view_as_windows(v, size)):
        e=0
        for pat in pat1:
            x =[x1+c, x2+e]
            y = [y1+c, y2+e]
            #print(x,y)
            a = abs(p - pat.mean())
            a = math.pow(a, 2)
            b = (size[0]*size[1])
            d = a/b
            dist[d] = [x, y]
            if(d < min_d):
                min_d = d
            e+=1
        c+=1 
    dist = collections.OrderedDict(sorted(dist.items()))
    dist = list(dist.values())
    return min_d,(dist[0:5])
    
def getPatch(u, shape):
    patch = np.split(u, [shape[0][0],shape[1][0]], axis=0)[1]
    patch = np.split(patch, [shape[0][1], shape[1][1]], axis=1)[1]
    return patch
    
def emission_func(labels, regions, u):
    phi = []
    for i in range(len(regions)):
        patch = getPatch(u, regions[0])
        size=patch.shape[0]*patch.shape[1]
        patch = patch.mean()
        dist_all_labels=[]
        for j in range(len(labels[i])):
            label = labels[i][j].mean()
            dist=abs(patch-label)/size
            dist_all_labels.append(dist)
        phi.append(dist_all_labels)
    return(phi)    
    
def main():
    threshold = 15
    l_d, l_s, l_r = 2,2,1
    u = cv2.imread('inputImg/input2.png')
    print(u.shape)
    v = cv2.imread('styleImg/exampleImg.png')
    img_yuv_U =  BGR2YUV(u)
    img_yuv_V =  BGR2YUV(v)
    start = (0,0)
    end = (img_yuv_U.shape[0], img_yuv_U.shape[1])
    ind = quadtree1(start, end)
    split_reg = []
    labels = []
    regions = []
    count = 0
    d1, d2 ={}, {}
    while(True):
        print(count)
        patch = getPatch(img_yuv_U, ind[0])
        dis, label = min_distance(patch, patch.shape, img_yuv_V)
        if(((( patch.std() + dis) > threshold) and (math.pow(patch.shape[0], 2) > 64)) or (math.pow(patch.shape[0], 2) > (256*256))) :
            print("Patch Divided")
            ind.extend(quadtree1(tuple(ind[0][0]), tuple(ind[0][1])))
        else:
            labels.append(label)
            split_reg.append(ind[0])
        regions.append(ind[0])
        ind.remove(ind[0])
        print("length ", len(ind))
        if not ind:
            print("its me1")
            break
        if(len(ind)==0):
            print("it's me2")
            break
        '''if(count==1):
            break'''
        count+=1
    d1["leaf"] = split_reg
    d2["label"] = labels
    json.dump(d1, open("leafs_input2.txt","w"))
    json.dump(d2, open("labels_input2.txt","w"))
    print(len(regions))
    for vertices in regions :
        #print((vertices[0][1],ve.rtices[0][0]),(vertices[1][1]-1,vertices[1][0]-1))
        x=cv2.rectangle(u,(vertices[0][1],vertices[0][0]),((vertices[1][1]),(vertices[1][0])),(0,0,0),1)
    cv2.imshow('Visualization',u)
    cv2.imwrite('splitPatches_ceramic.jpg',u)
    cv2.waitKey(0)   
    #phi = emission_func(labels, regions, img_yuv_U)
    #psi = transition_func
        
if __name__=="__main__":
       main()
