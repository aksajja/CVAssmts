import sys
import os
import numpy as np
from main import GCClient
import re
import cv2

osname = os.name

nargin = len(sys.argv)

if nargin < 2:
	raise ImportError('More args needed')
elif nargin == 2:
	imgroute = sys.argv[1]
	iteration_count = 1
	component_count = 5
elif nargin == 3:
	imgroute = sys.argv[1]
	iteration_count = sys.argv[2]
	component_count = 5
else:
	imgroute = sys.argv[1]
	iteration_count = sys.argv[2]
	component_count = sys.argv[3]

# if osname == 'nt':
imgname = re.findall(r'\w+\.\w+', imgroute)[0]
imgname = re.findall(r'^\w+', imgname)[0]
# elif osname == 'posix':
	# imgname = re.findall(r'^\S+\.', img[1:])[0][:-1]

if not os.path.isfile(imgroute):
	raise ImportError("Not a valid image")

try:
	img = cv2.imread(imgroute, cv2.IMREAD_COLOR)
except AttributeError:
	raise ImportError("Not a valid image")

output = np.zeros(img.shape, np.uint8)

GC = GCClient(img, component_count)
cv2.namedWindow('output')
cv2.namedWindow('input')
a = cv2.setMouseCallback('input', GC.init_mask)
cv2.moveWindow('input', img.shape[0]+100, img.shape[1]+100)

count = 0
print("Instructions: \n")
print("Draw a rectangle around the object using right mouse button \n")
print('Press N to continue \n')

while True:
	cv2.imshow('output', output)
	cv2.imshow('input', np.asarray(GC.img, dtype=np.uint8))
	k = 0xFF & cv2.waitKey(1)

	if k == 27:
		break
	elif k == ord('s'):
		cv2.imwrite('%s_gc.jpg'%(imgname), output)
	elif k == ord('n'):
		if GC.rect_or_mask == 0:
			GC.run()
			GC.rect_or_mask = 1
		elif GC.rect_or_mask == 1:
			GC.iter(1)
	
	elif k == ord('0'):
		GC._DRAW_VAL = GC._DRAW_BG

	elif k == ord('1'):
		GC._DRAW_VAL = GC._DRAW_FG

	elif k == ord('2'):
		GC._DRAW_VAL = GC._DRAW_PR_BG

	elif k == ord('3'):
		GC._DRAW_VAL = GC._DRAW_PR_FG

	elif k == ord('r'):
		GC.__init__(img, component_count)
	FGD = np.where((GC._mask == 1) + (GC._mask == 3), 255, 0).astype('uint8')
	output = cv2.bitwise_and(GC.img2, GC.img2, mask = FGD)
cv2.destroyAllWindows()
