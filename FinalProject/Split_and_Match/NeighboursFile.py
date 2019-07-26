import json
import numpy as np
import itertools


d2 = json.load(open("text.txt"))
all_leaf_patches=list(itertools.chain(*d2.values()))
print('# of Leaf Nodes  in Source Image: ',len(all_leaf_patches))

#exit()

############Finding Neighbouring for given PATCH #############

def Neighbouring_patches(MainPatch):
	right_sides=[]
	left_sides=[]
	up_sides=[]
	down_sides=[]

	for neighbours_patch in all_leaf_patches:
		#RightSide
		if((MainPatch[1][1]==neighbours_patch[0][1] or MainPatch[1][1]+1==neighbours_patch[0][1]) and not (neighbours_patch[0][0]<MainPatch[0][0] 				and neighbours_patch[1][0]<MainPatch[1][0] or neighbours_patch[0][0]>MainPatch[1][0] and neighbours_patch[1][0]>MainPatch[1][0])):
			right_sides.append(neighbours_patch)

		##Downside
		if((MainPatch[1][0]==neighbours_patch[0][0] or MainPatch[1][0]+1==neighbours_patch[0][0]) and not (neighbours_patch[0][1]<MainPatch[0][1] 				and neighbours_patch[1][1]<MainPatch[1][1] or neighbours_patch[0][1]>MainPatch[1][1] and neighbours_patch[1][1]>MainPatch[1][1])):
			down_sides.append(neighbours_patch)
			
		##LeftSide
		if((MainPatch[0][1]==neighbours_patch[1][1] or MainPatch[0][1]-1==neighbours_patch[1][1]) and not (neighbours_patch[0][0]<MainPatch[0][0] 				and neighbours_patch[1][0]<MainPatch[1][0] or neighbours_patch[0][0]>MainPatch[1][0] and neighbours_patch[1][0]>MainPatch[1][0])):
			left_sides.append(neighbours_patch)

		##UpSide
		if((MainPatch[0][0]==neighbours_patch[1][0] or MainPatch[0][0]-1==neighbours_patch[1][0]) and not (neighbours_patch[0][1]<MainPatch[0][1] 				and neighbours_patch[1][1]<MainPatch[1][1] or neighbours_patch[0][1]>MainPatch[1][1] and neighbours_patch[1][1]>MainPatch[1][1])):
			up_sides.append(neighbours_patch)

	return right_sides,left_sides,up_sides,down_sides



Neighbours_dict={}
l=[]
for n,MainPatch in enumerate(all_leaf_patches):
	right_sides,left_sides,up_sides,down_sides=Neighbouring_patches(MainPatch)
	four_neighbours_combined=[right_sides,left_sides,up_sides,down_sides]
	#print("#####################################################",n+1)
	four_neighbours_combined=list(itertools.chain(*four_neighbours_combined))

	'''##Cross check patche
	if(n==300):
		print(MainPatch)
		print(four_neighbours_combined,len(four_neighbours_combined))

	l.append(len(four_neighbours_combined))
	#print("\n\n")
	'''

	Neighbours_dict[str(MainPatch)]=four_neighbours_combined

json.dump(Neighbours_dict, open("Neighbours_dict_text.txt",'w'))
print('All neighbouring_dict dumping done...\n')


