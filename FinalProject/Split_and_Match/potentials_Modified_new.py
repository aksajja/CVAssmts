import collections
import numpy as np
import cv2 
from functools import reduce
from sklearn.feature_extraction.image import extract_patches_2d
import warnings
warnings.filterwarnings('ignore')
import math
from math import sqrt
import json
from ast import literal_eval

#import NeighboursFile
#import Adaptive_parition

############## Pre-processing #################
original_img='U.jpeg'
example_img='v.png'
#example_img='V.jpeg'

o_img = cv2.imread(original_img)
clone_u = o_img.copy()
e_img =	cv2.imread(example_img)
img_yuv_U =  cv2.cvtColor(o_img, cv2.COLOR_BGR2YUV)
img_yuv_V =  cv2.cvtColor(e_img, cv2.COLOR_BGR2YUV)

y, u, v = cv2.split(img_yuv_U)
y2, u2, v2 = cv2.split(img_yuv_V)

dum_V = y2
dum_U= y 

image_matrix=y
style_matrix=y2

# CHANGED : image size may not be square, hence separate out of bounds check
img_size_rows=y.shape[0]
img_size_columns=y.shape[1]

# CHANGED : style size may not be square, hence separate out of bounds check
style_size_rows=y.shape[0]
style_size_columns=y.shape[1]



#labels=Adaptive_parition.lb

dl = json.load(open("labels.txt"))
#labels=list(itertools.chain(*dl.values()))
labels=dl

dn=json.load(open('Neighbours_dict_text.txt'))
neighbour_dict=dn
#neighbour_dict=NeighboursFile.Neighbours_dict
print("label_dict: ",len(labels))
print("neighbour_dict",len(neighbour_dict))
print("\n")

#################### Main snippet Program ########################

# global labels={region:[10 labels]}, neighbour_dict={region1:[neighbour1, ... ], region2 : ...., ... }), 
# binaries={ {[patch1, patch2] : {[label1, label2] : norm, .. }, ... } }

import numpy as np
import cv2
#global labels, neighbour_dict, img_size, image_matrix, style_matrixTypeError: unhashable type: 'list'


def l2_norm_unary (patch1, patch2) :
	patch1=literal_eval(patch1)
	patch_matrix1=np.split(image_matrix,[patch1[0][0],patch1[1][0]],axis=0)[1]  ##X-axis
	patch_matrix1=np.split(patch_matrix1,[patch1[0][1],patch1[1][1]],axis=1)[1] ##Y-axis
	patch_matrix2=np.split(style_matrix,[patch2[0][0],patch2[1][0]],axis=0)[1]  ##X-axis
	patch_matrix2=np.split(patch_matrix2,[patch2[0][1],patch2[1][1]],axis=1)[1] ##Y-axis
	norm_val = np.sum((patch_matrix1-patch_matrix2)**2)/(patch_matrix1.shape[0]**2)
	#print('norm_val : ',norm_val)
	return norm_val

def l2_norm_binary (patch1, patch2) :
	print('l2_norm_binary ....\n')
	print('image shape: ', image_matrix.shape,' style shape: ', style_matrix.shape)
	print('patch1, patch2 : ',patch1,patch2)
	print("\n")
	# CHANGED : image_matrix to style_matrix
	patch_matrix1=np.split(style_matrix,[patch1[0][0],patch1[1][0]],axis=0)[1]  ##X-axis
	patch_matrix1=np.split(patch_matrix1,[patch1[0][1],patch1[1][1]],axis=1)[1] ##Y-axis

	patch_matrix2=np.split(style_matrix,[patch2[0][0],patch2[1][0]],axis=0)[1]  ##X-axis
	patch_matrix2=np.split(patch_matrix2,[patch2[0][1],patch2[1][1]],axis=1)[1] ##Y-axis
	print('patch_matrix1 - patch_matrix2 Shapes : ',patch_matrix1.shape, patch_matrix2.shape)
	print("\n\n")
	norm_val = np.sum((patch_matrix1-patch_matrix2)**2)/(patch_matrix1.shape[0]**2)
	return norm_val
    

def extended_label(label, extension) :
	label = np.asarray(label)
	extension = np.asarray(extension)

	# CHANGED : extension to be done only wrt vertex1 of label; Out of bounds check added
	shaded = extension+label[0]
	if shaded[1][0]>style_size_rows-1 or shaded[1][1]>style_size_columns-1 :
		return None
	if shaded[0][0]<0 or shaded[0][1]<0 :
		return None
	
	return list(shaded)


def binary_pots(reg1, reg2, extension1, extension2):
	print('binary_pots ....\n')
	dict_label_potentials = {}
	for label1 in labels[str(reg1)] :
		for label2 in labels[str(reg2)] :
			patch1 = extended_label(label1, extension1)
			patch2 = extended_label(label2, extension2)
			print("patch1, patch2 : ",patch1, patch2)
			if patch1 is None or patch2 is None :
				dict_label_potentials[str([label1,label2])] = None
			else :
				dict_label_potentials[str([label1,label2])] = l2_norm_binary(patch1, patch2)
			print("---------------------------\n")
	return dict_label_potentials


def calculate_unaries(labels_dict):
	# store as dict(), dict1 = {region[i]:{label[j]: unary, ...}}
	unaries = {}
	for partition in labels_dict :
		for label in labels_dict[partition] :
			if partition not in unaries :
				unaries[partition] = {}
			unaries[partition][str(label)]= l2_norm_unary(partition, label)
	return unaries

def extend_region(patch):
	# check for out of bounds
	ver1, ver3 = patch
	Ti = ver3[0]-ver1[0]
	Ti_2 = int(Ti/2)

	# Check if index is less than side-1 (subtract as index starts at 0), set to 0.
	if ver1[0]<Ti_2-1 :
	    extended_x1 = 0
	else :
	    extended_x1 = ver1[0]-(Ti_2)
	
	if ver1[1]<Ti_2-1 :
	    extended_y1 = 0
	else :
	    extended_y1 = ver1[1]-(Ti_2)

	if ver3[0]+Ti_2 > img_size_rows-1 :
	    extended_x2 = img_size_rows-1
	else :
	    extended_x2 = ver3[0]+(Ti_2)

	if ver3[1]+Ti_2 > img_size_columns-1 :
	    extended_y2 = img_size_columns-1
	else :
	    extended_y2 = ver3[1]+(Ti_2)

	return [extended_x1,extended_y1],[extended_x2,extended_y2]

def find_intersection_region(reg1, reg2) :
	# Which corner is inside the other squre. Do for both.
	# One region's x & y coordinates which lie within the other regions xs and ys.
	#final_x1,final_x2,final_y1,final_y2
	reg1_ver1, reg1_ver2 = reg1
	reg1_x1, reg1_x2 = reg1_ver1[0],reg1_ver2[0]
	reg1_y1, reg1_y2 = reg1_ver1[1],reg1_ver2[1]
	reg2_ver1, reg2_ver2 = reg2
	reg2_x1, reg2_x2 = reg2_ver1[0],reg2_ver2[0]
	reg2_y1, reg2_y2 = reg2_ver1[1],reg2_ver2[1]
	# check if reg1 in reg2
	if reg1_x1 in range(reg2_x1, reg2_x2+1) :
		final_x1 = reg1_x1 
	if reg1_x2 in range(reg2_x1, reg2_x2+1) :
		final_x2 = reg1_x2
	if reg1_y1 in range(reg2_y1, reg2_y2+1) :
		final_y1 = reg1_y1 
	if reg1_y2 in range(reg2_y1, reg2_y2+1) :
		final_y2 = reg1_y2
	# check if reg2 in reg1 
	if reg2_x1 in range(reg1_x1, reg1_x2+1) :
		final_x1 = reg2_x1 
	if reg2_x2 in range(reg1_x1, reg1_x2+1) :
		final_x2 = reg2_x2 
	if reg2_y1 in range(reg1_y1, reg1_y2+1) :
		final_y1 = reg2_y1 
	if reg2_y2 in range(reg1_y1, reg1_y2+1) :
		final_y2 = reg2_y2
	print('find_intersection_region Final Matrix : ',[final_x1,final_y1],[final_x2,final_y2])
	return [final_x1,final_y1],[final_x2,final_y2]

def transform_wrt_origin(points, origin):
	origin = np.asarray(origin)
	points = np.asarray(points)
	transformed_points=(list(points - origin))

	return transformed_points

def calculate_binaries():
	#for each pair findIntersection
	binaries = {}
	for region in neighbour_dict :
		reg1 = literal_eval(region)
		print('reg1',reg1)
		extended_reg1 = extend_region(reg1)
		print('extended_reg1 : ',extended_reg1)
		origin1 = reg1[0]
		for neighbour in neighbour_dict[region] :
			reg2 = neighbour
			print("reg2: ",reg2)
			origin2 = reg2[0]
			#extend region & neighbour. Handle edge cases
			extended_reg2 = extend_region(reg2)
			print('extended_reg2 : ',extended_reg2)
			print("\n\n")
			intersection = find_intersection_region(extended_reg1, extended_reg2)
			transformed_intersection_o1 = transform_wrt_origin(intersection, origin1)
			transformed_intersection_o2 = transform_wrt_origin(intersection, origin2)
			# dict2 = {[reg1,reg2] : {[label1x, label2y] : pair_pot, ... } ... }
			print('transformed_intersection_o1 : ',transformed_intersection_o1)
			print('transformed_intersection_o2 : ',transformed_intersection_o2)
			print("\n")
			binaries[str([reg1,reg2])]= binary_pots(reg1, reg2, transformed_intersection_o1, transformed_intersection_o2)
	return binaries


#### Calling Function:
unary_dict = calculate_unaries(labels)
print("Unary Potentials completed...")
print("\n########################################\n")
# CHANGED : IMPORTANT NOTE - Certain label pairs of regions may be NONE because of out of bounds issue. Disregard them.
binary_dict = calculate_binaries()
print("\n########################################\n")
