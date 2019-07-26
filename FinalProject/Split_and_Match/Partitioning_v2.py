import collections
import numpy as np
import cv2 
from functools import reduce
from sklearn.feature_extraction.image import extract_patches_2d
import warnings
warnings.filterwarnings('ignore')
import math
from math import sqrt



r0=8*8
r1=256*256
threshold=15


original_img='sourceImg.png'
example_img='exampleImg.png'
o_img = cv2.imread(original_img)
clone_u = o_img.copy()
e_img =	cv2.imread(example_img)
img_yuv_U =  cv2.cvtColor(o_img, cv2.COLOR_BGR2YUV)
img_yuv_V =  cv2.cvtColor(e_img, cv2.COLOR_BGR2YUV)

y, u, v = cv2.split(img_yuv_U)
y2, u2, v2 = cv2.split(img_yuv_V)

dum_V=y2[:200][:,:200]
dum_U=y[:200][:,:200]
# dum_V = y2
# dum_U = y

def quardtree(region_size):
	patch_size=(int(region_size[0]/2),int(region_size[1]/2))
	return patch_size	

def distance_norm(Pxi, Py,size):
	num=(Pxi-Py)*2
	d=num/(size.shape[0]*size.shape[1])	
	return d

def patch_division(patch_size,patch):
	box1=patch[:patch_size[0]][:,:patch_size[1]]
	box2=patch[:patch_size[0]][:,patch_size[1]:]

	box3=patch[patch_size[0]:][:,:patch_size[1]]
	box4=patch[patch_size[0]:][:,patch_size[1]:]
	return box1,box2,box3,box4

def patch_division_2(vertex1, vertex3):
	x = vertex3[0]-vertex1[0] -1
	print(x)
	firstSide = int(x/2)
	remainingSide = int(x - firstSide)
	box1=[vertex1,[vertex1[0]+firstSide,vertex1[1]+firstSide]]

	box2=[[vertex1[0],vertex1[1]+firstSide+1],[vertex1[0]+remainingSide,vertex3[1]]]

	box3=[[vertex3[0]-firstSide,vertex3[1]-firstSide],vertex3]

	box4=[[vertex1[0]+firstSide+1,vertex1[1]],[vertex3[0],vertex1[1]+remainingSide]]
	return box1,box2,box3,box4


def visualize(matrix,position):
	split_1=np.split(matrix,[position[0][0],position[1][0]],axis=0)[1]  ##X-axis
	split_2=np.split(split_1,[position[0][1],position[1][1]],axis=1)[1] ##Y-axis
	return split_2


##Initially....
patch_size= quardtree(dum_U.shape)
print(patch_size)
box1,box2,box3,box4=patch_division_2([0,0],[dum_U.shape[0]-1,dum_U.shape[1]-1])

print([0,0],[dum_U.shape[0]-1,dum_U.shape[1]-1])

all_regions=[box1,box2,box3,box4] #all patches positions

print("all_regions : ",all_regions)
possible_patches= extract_patches_2d(dum_V, patch_size) ## In V:


nodes_dict={}
def parition_concept(Patch_position,patchNO,Each_Patch_Size,std_dev,LEAF_NODES):
	Patch=visualize(dum_U,Patch_position)
	possible_patches= extract_patches_2d(dum_V, Each_Patch_Size)
	Ti=Patch.shape
	Ti=reduce(lambda x, y: x*y, list(Ti))
	temp_patch_array = [int(Each_Patch_Size[0]/2),int(Each_Patch_Size[1]/2)]
	center_position=np.array(temp_patch_array)
	center_position.astype(int)
	U_coordinates=Patch[int(Each_Patch_Size[0]/2)][int(Each_Patch_Size[1]/2)]
	min_val=9999999999
	min_ind=0
	for i in range(len(possible_patches)):
		center_position_V=np.array(list(possible_patches[i].shape))/2
		V_coordinates=possible_patches[i][center_position[0]][center_position[1]]
		similarity=distance_norm(U_coordinates,V_coordinates,possible_patches[i])
		if(min_val>similarity):	
			min_val=similarity
			min_ind=i
	split_condition=std_dev+min_val
	#print('min_val : ',min_val,min_ind,possible_patches[min_ind])
	if((split_condition>threshold and Ti>r0) or Ti>r1):
		print("Split further....\n")
		print(Patch_position)
		size= quardtree(Patch.shape)
		#box1,box2,box3,box4=patch_division_2(size,Patch) ###DOUBT
		box1,box2,box3,box4=patch_division_2(Patch_position[0],Patch_position[1]) ###DOUBT
		new_regions=[box1,box2,box3,box4]
		print("new_regions: ",new_regions)
		for rep_ri in range(4):
			PATCH=visualize(dum_U,new_regions[rep_ri])
			std_dev=np.std(PATCH)
			parition_concept(new_regions[rep_ri],rep_ri+1,size,std_dev,LEAF_NODES)
	else:
		print("no spliting further....\n")
		LEAF_NODES.append(Patch_position)
	return LEAF_NODES


def find_labels(pa,dum_U,dum_V,img_yuv_V):
	K=10
	slide=2

	# For a single patch in U
	# Since it's a square, sides along both axes is same.
	patch_size = [pa[1][0]-pa[0][0],pa[1][0]-pa[0][0]]
	Ti = patch_size[0]
	xi_rows = np.split(dum_U,[pa[0][0],pa[1][0]],axis=0)[1]
	xi = np.split(xi_rows,[pa[0][1],pa[1][1]],axis=1)[1]

	# xi = pa
	# Variables for start and end vertices of patch in 'V'
	start_x = 0
	start_y = 0
	end_x = patch_size[0]
	end_y = patch_size[1]

	xi_norms = dict()
	while end_y<dum_V.shape[1]+1 :
		yi = np.split(dum_V,[start_x,end_x],axis=0)[1]
		yi = np.split(yi,[start_y,end_y],axis=1)[1]
		l2 = np.linalg.norm(xi-yi)
		# Dictionary dist is key and vertices are values
		# We assume L2 norm values are unique
		xi_norms[l2]=[[start_x,start_y],[end_x,end_y]]
		start_x+=slide
		end_x+=slide
		if end_x>dum_V.shape[0]:
			start_y+=slide
			end_y+=slide
			start_x=0
			end_x=patch_size[0]

	xi_norms = collections.OrderedDict(sorted(xi_norms.items()))
	# Pick K from sorted list
	K_centers = dict()
	i = 1
	for key in xi_norms:
		# Center = [x1+x0/2,y1+y0/2]
		shouldContinue = False
		center = [(xi_norms[key][1][0]+xi_norms[key][0][0])/2, (xi_norms[key][1][1]+xi_norms[key][0][1])/2]
		for mid in K_centers:
			doc = np.linalg.norm(np.asarray(center) - np.asarray(K_centers[mid]))
			if doc < Ti/2 :
				# print("Dragon food")
				shouldContinue = True
				break
		if shouldContinue :
			continue
		K_centers[key]=center

		if len(K_centers)>=K:
			break
		
	# K_labels = []
	# print(K_centers)

	# We match the keys in K_centers (contain centers) with xi_norms (contain vertices)
	# Rectangles are drawn on the image for visualization
	temp_v = img_yuv_V.copy()
	temp_u = clone_u.copy()
	for key in K_centers :
		vertices = xi_norms[key]
		cv2.rectangle(temp_v,(vertices[0][0],vertices[0][1]),(vertices[1][0],vertices[1][1]),(0,0,255),1)
	cv2.rectangle(temp_u,(int(pa[0][1]),int(pa[0][0])),(int(pa[1][1]),int(pa[1][0])),(0,0,255),1)
	cv2.imshow('VisualizationV',temp_v)
	cv2.imshow('VisualizationU',temp_u)
	cv2.imwrite('./CV__Project/code/visual1.jpg',temp_v)
	cv2.waitKey(0)


for ri in range(4):
	print('Ri :',ri)
	LEAF_NODES=[]
	PATCH=visualize(dum_U,all_regions[ri])
	std_dev=np.std(PATCH)
	LEAF_NODES = parition_concept(all_regions[ri],ri,PATCH.shape,std_dev, LEAF_NODES)
	nodes_dict[ri]=LEAF_NODES
	#break

print("\n\n@@@@@@@@@@@@@@@@@@@@ Leaf Nodes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
print(nodes_dict)

for nodes in nodes_dict :
	for vertices in nodes_dict[nodes] :
		# print(key)
		# for vertices in LEAF_NODES[key] :
		print(vertices, "\n")
		cv2.rectangle(o_img,(vertices[0][1],vertices[0][0]),(vertices[1][1],vertices[1][0]),(0,0,255),1)

for vertices in all_regions :
    cv2.rectangle(o_img,(vertices[0][1],vertices[0][0]),(vertices[1][1],vertices[1][0]),(0,0,255),1)

cv2.imshow('Visualization',o_img)
cv2.imwrite('splitPatches.jpg',o_img)
cv2.waitKey(0)

for key,value in nodes_dict.items():
	for pa in value:
		# print(pa)
		find_labels(pa,dum_U,dum_V,e_img)
		print("\n")
		
	print("\n--------------------------------------------------------------\n")

