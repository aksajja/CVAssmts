import cv2
import json
import math
import numpy as np
import itertools

def getPatch(u, shape):
    patch = np.split(u, [shape[0][0],shape[1][0]], axis=0)[1]
    patch = np.split(patch, [shape[0][1], shape[1][1]], axis=1)[1]
    return patch

def BGR2YUV(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    
    y, u, v = cv2.split(img)
    return y
        
def hmm(leaf, label):
    e =getPatch(u,leaf).mean()
    min = 9999999
    for i in range(2):
        s = getPatch(v, label[i])
        e1 = s.mean()
        for j in range(2):
            if(i!=j):
                e2 = getPatch(v, label[j]).mean()
                dis = e1 - e2
                dis = math.pow(dis, 2) / math.pow(s.shape[0], 2)
                if(min>dis):
                    min = dis
                    temp = j
    return temp
    
def MRF(leaf, label):
    R =getPatch(u1,leaf)
    R_i = R.mean()
    min = 999999
    tran_p =0
    for i in range(2):
        L = getPatch(v, label[i])
        L_i = L.mean()
        for j in range(2):
            if(i!=j):
                L_j = getPatch(v, label[j]).mean()
                d1 = L_i - L_j
                tran_p = tran_p * (math.pow(d1, 2) / math.pow(L.shape[0], 2))
        d2 = R_i - L_i
        emis_p = math.pow(d2, 2) / math.pow(L.shape[0], 2)
        Reg_label =emis_p * tran_p
        if(min>Reg_label):
            min = Reg_label
            temp = i 
    return temp            
         
u1 = cv2.imread('inputImg/person2.jpg')
u =  BGR2YUV(u1)
v1 = cv2.imread('styleImg/style1.png')
v = BGR2YUV(v1)
d1 = json.load(open("txt/leafs_llama.txt"))
all_leaf_patches=list(itertools.chain(*d1.values()))

d2 = json.load(open("txt/labels_llama.txt"))
all_label_patches=list(itertools.chain(*d2.values()))


for i in range(len(all_leaf_patches)):
    l1 = all_leaf_patches[i]
    l2 = all_label_patches[i]
    t = MRF(l1, l2)
    t = getPatch(v1, l2[t])
    t1 = getPatch(u1, l1)
    t = np.add(t,t1)
    u1[l1[0][0]:l1[1][0], l1[0][1]:l1[1][1]] = t

cv2.imshow('Visualization',u1)
cv2.imwrite("texture_llama.jpg",u1)
cv2.waitKey(0) 
