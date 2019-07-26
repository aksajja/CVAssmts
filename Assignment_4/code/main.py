import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import random
from k_means import kmeans
from gcgraph import GCGraph
import time

def timeit(func):
	def wrapper(*args, **kw):
		time1 = time.time()
		result = func(*args, **kw)
		time2 = time.time()
		return result
	return wrapper

def get_size(img):
	return list(img.shape)[:2]

def flat(img):
	return img.reshape([1, img.size])[0]

class GMM:
	'''The GMM: Gaussian Mixture Model algorithm'''
	'''Each point in the image belongs to a GMM, and because each pixel owns
		three channels: RGB, so each component owns three means, 9 covs and a weight.'''
	
	def __init__(self, k = 5):
		'''k is the number of components of GMM'''
		self.k = k
		self.weights = np.asarray([0. for i in range(k)]) # Weight of each component
		self.means = np.asarray([[0., 0., 0.] for i in range(k)]) # Means of each component
		self.covs = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)]) # Covs of each component
		self.cov_inv = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])
		self.cov_det = np.asarray([0. for i in range(k)])
		self.pixel_counts = np.asarray([0. for i in range(k)]) # Count of pixels in each components
		self.pixel_total_count = 0 # The total number of pixels in the GMM
		
		# The following two parameters are assistant parameters for counting pixels and calc. pars.
		self._sums = np.asarray([[0., 0., 0.] for i in range(k)])
		self._prods = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])

	def _prob_pixel_component(self, pixel, ci):
		inv = self.cov_inv[ci]
		det = self.cov_det[ci]
		t = pixel - self.means[ci]
		nt = np.asarray([t])
		mult = np.dot(inv, np.transpose(nt))
		mult = np.dot(nt, mult)
		return (1/np.sqrt(det) * np.exp(-0.5*mult))[0][0]

	def prob_pixel_GMM(self, pixel):	
		return sum([self._prob_pixel_component(pixel, ci) * self.weights[ci] for ci in range(5)])

	def most_likely_pixel_component(self, pixel):
		prob = np.asarray([self._prob_pixel_component(pixel, ci) for ci in range(5)])
		return prob.argmax(0)

	def add_pixel(self, pixel, ci):
		tp = pixel.copy().astype(np.float32)
		self._sums[ci] += tp
		tp.shape = (tp.size, 1)
		self._prods[ci] += np.dot(tp, np.transpose(tp))
		self.pixel_counts[ci] += 1
		self.pixel_total_count += 1

	def __learning(self):
		variance = 0.01
		zeros = np.where(np.asarray(self.pixel_counts) == 0)
		notzeros = np.where(np.asarray(self.pixel_counts) != 0)
		self.weights = np.asarray([self.pixel_counts[i]/self.pixel_total_count for i in range(5)]) # The weight of each comp. is the pixels in the comp. / total pixels.
		self.means = np.asarray([self._sums[i]/self.pixel_counts[i] for i in range(5)]) # The mean of each comp. is the sum of pixels of the comp. / the number of pixels in the comp.
		nm = np.asarray([[i] for i in self.means])
		self.covs = np.asarray([self._prods[i]/self.pixel_counts[i] - np.dot(np.transpose(nm[i]), nm[i]) for i in range(5)]) # The cov of each comp.
		self.cov_det = np.asarray([np.linalg.det(cov) for cov in self.covs])
		for i in range(5):
			while self.cov_det[i] <= 0:
				self.covs[i] += np.diag([variance for i in range(3)])
				self.cov_det[i] = np.linalg.det(self.covs[i])
		self.cov_inv = np.asarray([np.linalg.inv(cov) for cov in self.covs])

	def learning(self):
		variance = 0.01
		for ci in range(5):
			n = self.pixel_counts[ci]
			if n == 0:
				self.weights[ci] = 0
			else:
				self.weights[ci] = n/self.pixel_total_count
				self.means[ci] = self._sums[ci]/n
				nm = self.means[ci].copy()
				nm.shape = (nm.size, 1)
				self.covs[ci] = self._prods[ci]/n - np.dot(nm, np.transpose(nm))
				self.cov_det[ci] = np.linalg.det(self.covs[ci])
			while self.cov_det[ci] <= 0:
				self.covs[ci] += np.diag([variance for i in range(3)])
				self.cov_det[ci] = np.linalg.det(self.covs[ci])
			self.cov_inv[ci] = np.linalg.inv(self.covs[ci])


class GCClient:
	def __init__(self, img, k):
		self.k = k # The number of components in each GMM model
		self.img = np.asarray(img, dtype = np.float32)
		self.img2 = img
		self.rows, self.cols = get_size(img)
		self.gamma = 50
		self.lam = 9*self.gamma
		self.beta = 0
		self._BLUE = [255,0,0]        # rectangle color
		self._RED = [0,0,255]         # PR BG
		self._GREEN = [0,255,0]       # PR FG
		self._BLACK = [0,0,0]         # sure BG
		self._WHITE = [255,255,255]   # sure FG
		self._DRAW_BG = {'color':self._BLACK, 'val':0}
		self._DRAW_FG = {'color':self._WHITE, 'val':1}
		self._DRAW_PR_FG = {'color':self._GREEN, 'val':3}
		self._DRAW_PR_BG = {'color':self._RED, 'val':2}
		self._rect = [0, 0, 1, 1]
		self._drawing = False         # flag for drawing curves
		self._rectangle = False       # flag for drawing rect
		self._rect_over = False       # flag to check if rect drawn
		self._thickness = 5           # brush thickness
		self._GC_BGD = 0	#{'color' : BLACK, 'val' : 0}
		self._GC_FGD = 1	#{'color' : WHITE, 'val' : 1}
		self._GC_PR_BGD = 2	#{'color' : GREEN, 'val' : 3}
		self._GC_PR_FGD = 3	#{'color' : RED, 'val' : 2}
		self.calc_beta()
		self.calc_nearby_weight()
		self._DRAW_VAL = None
		self._mask = np.zeros([self.rows, self.cols], dtype = np.uint8) # Init the mask
		self._mask[:, :] = self._GC_BGD

	def calc_beta(self):
		beta = 0
		self._left_diff = self.img[:, 1:] - self.img[:, :-1] # Left-difference
		self._upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1] # Up-Left difference
		self._up_diff = self.img[1:, :] - self.img[:-1, :] # Up-difference
		self._upright_diff = self.img[1:, :-1] - self.img[:-1, 1:] # Up-Right difference
		beta = (self._left_diff*self._left_diff).sum() + (self._upleft_diff*self._upleft_diff).sum() \
			+ (self._up_diff*self._up_diff).sum() + (self._upright_diff*self._upright_diff).sum() # According to the formula
		self.beta = 1/(2*beta/(4*self.cols*self.rows - 3*self.cols - 3*self.rows + 2))

	def calc_nearby_weight(self):
		self.left_weight = np.zeros([self.rows, self.cols])
		self.upleft_weight = np.zeros([self.rows, self.cols])
		self.up_weight = np.zeros([self.rows, self.cols])
		self.upright_weight = np.zeros([self.rows, self.cols])
		for y in range(self.rows):
			for x in range(self.cols):
				color = self.img[y, x]
				if x >= 1:
					diff = color - self.img[y, x-1]
					self.left_weight[y, x] = self.gamma*np.exp(-self.beta*(diff*diff).sum())
				if x >= 1 and y >= 1:
					diff = color - self.img[y-1, x-1]
					self.upleft_weight[y, x] = self.gamma/np.sqrt(2) * np.exp(-self.beta*(diff*diff).sum())
				if y >= 1:
					diff = color - self.img[y-1, x]
					self.up_weight[y, x] = self.gamma*np.exp(-self.beta*(diff*diff).sum())
				if x+1 < self.cols and y >= 1:
					diff = color - self.img[y-1, x+1]
					self.upright_weight[y, x] = self.gamma/np.sqrt(2)*np.exp(-self.beta*(diff*diff).sum())
		
	def init_mask(self, event, x, y, flags, param):
		if event == cv2.EVENT_RBUTTONDOWN:
			self._rectangle = True
			self._ix,self._iy = x,y

		elif event == cv2.EVENT_MOUSEMOVE:
		    if self._rectangle == True:
		    	self.img = self.img2.copy()
		    	cv2.rectangle(self.img,(self._ix,self._iy),(x,y),self._BLUE,2)
		    	self._rect = [min(self._ix,x),min(self._iy,y),abs(self._ix-x),abs(self._iy-y)]
		    	self.rect_or_mask = 0

		elif event == cv2.EVENT_RBUTTONUP:
			self._rectangle = False
			self._rect_over = True
			cv2.rectangle(self.img,(self._ix,self._iy),(x,y),self._BLUE,2)
			self._rect = [min(self._ix,x),min(self._iy,y),abs(self._ix-x),abs(self._iy-y)]
			self.rect_or_mask = 0
			self._mask[self._rect[1]+self._thickness:self._rect[1]+self._rect[3]-self._thickness, self._rect[0]+self._thickness:self._rect[0]+self._rect[2]-self._thickness] = self._GC_PR_FGD

		if event == cv2.EVENT_LBUTTONDOWN:
			if self._rect_over == False:
			    print("Continue")
			else:
				self._drawing == True
				cv2.circle(self.img, (x, y), self._thickness, self._DRAW_VAL['color'], -1)
				cv2.circle(self._mask, (x, y), self._thickness, self._DRAW_VAL['val'], -1)

		elif event == cv2.EVENT_MOUSEMOVE:
			if self._drawing == True:
				cv2.circle(self.img, (x, y), self._thickness, self._DRAW_VAL['color'], -1)
				cv2.circle(self._mask, (x, y), self._thickness, self._DRAW_VAL['val'], -1)

		elif event == cv2.EVENT_LBUTTONUP:
			if self._drawing == True:
				self._drawing = False
				cv2.circle(self.img, (x, y), self._thickness, self._DRAW_VAL['color'], -1)
				cv2.circle(self._mask, (x, y), self._thickness, self._DRAW_VAL['val'], -1)

	def init_with_kmeans(self):
		print(self.cols*self.rows)
		print(len(list(np.where(self._mask == 0))[1]))
		max_iter = 2 # Max-iteration count for Kmeans
		self._bgd = np.where(np.logical_or(self._mask == self._GC_BGD, self._mask == self._GC_PR_BGD)) # Find the places where pixels in the mask MAY belong to BGD.
		self._fgd = np.where(np.logical_or(self._mask == self._GC_FGD, self._mask == self._GC_PR_FGD)) # Find the places where pixels in the mask MAY belong to FGD.
		self._BGDpixels = self.img[self._bgd]
		self._FGDpixels = self.img[self._fgd]
		KMB = kmeans(self._BGDpixels, dim = 3, n = 5, max_iter = max_iter) # The Background Model by kmeans
		KMF = kmeans(self._FGDpixels, dim = 3, n = 5, max_iter = max_iter) # The Foreground Model by kmeans
		KMB.run()
		KMF.run()
		self._BGD_by_components = KMB.output()
		self._FGD_by_components = KMF.output()
		self.BGD_GMM = GMM() # The GMM Model for BGD
		self.FGD_GMM = GMM() # The GMM Model for FGD
		for ci in range(5):
			for pixel in self._BGD_by_components[ci]:
				self.BGD_GMM.add_pixel(pixel, ci)
			for pixel in self._FGD_by_components[ci]:
				self.FGD_GMM.add_pixel(pixel, ci)
		self.BGD_GMM.learning()
		self.FGD_GMM.learning()

	def assign_GMM_components(self):
		self.components_index = np.zeros([self.rows, self.cols], dtype = np.uint)
		for y in range(self.rows):
			for x in range(self.cols):
				pixel = self.img[y, x]
				self.components_index[y, x] = self.BGD_GMM.most_likely_pixel_component(pixel) if (self._mask[y, x] \
					== self._GC_BGD or self._mask[y, x] == self._GC_PR_BGD) else self.FGD_GMM.most_likely_pixel_component(pixel)

	def _assign_GMM_components(self):
		self.components_index = np.zeros([self.rows, self.cols], dtype = np.uint)
		self.components_index[self._bgd] = [i[0] for i in self.BGD_GMM.vec_pix_comp(self.img[self._bgd])]
		self.components_index[self._fgd] = [i[0] for i in self.FGD_GMM.vec_pix_comp(self.img[self._fgd])]

	def learn_GMM_parameters(self):
		for ci in range(5):
			bgd_ci = np.where(np.logical_and(self.components_index == ci, np.logical_or(self._mask == self._GC_BGD, self._mask == self._GC_PR_BGD)))
			fgd_ci = np.where(np.logical_and(self.components_index == ci, np.logical_or(self._mask == self._GC_FGD, self._mask == self._GC_PR_FGD)))
			for pixel in self.img[bgd_ci]:
				self.BGD_GMM.add_pixel(pixel, ci)
			for pixel in self.img[fgd_ci]:
				self.FGD_GMM.add_pixel(pixel, ci)
		self.BGD_GMM.learning()
		self.FGD_GMM.learning()

	def construct_gcgraph(self, lam):
		vertex_count = self.cols*self.rows
		edge_count = 2*(4*vertex_count - 3*(self.rows + self.cols) + 2)
		self.graph = GCGraph(vertex_count, edge_count)
		for y in range(self.rows):
			for x in range(self.cols):
				vertex_index = self.graph.add_vertex() # add-node and return its index
				color = self.img[y, x]
				if self._mask[y, x] == self._GC_PR_BGD or self._mask[y, x] == self._GC_PR_FGD:
					fromSource = -np.log(self.BGD_GMM.prob_pixel_GMM(color))
					toSink = -np.log(self.FGD_GMM.prob_pixel_GMM(color))
				elif self._mask[y, x] == self._GC_BGD:
					fromSource = 0
					toSink = lam
				else:
					fromSource = lam
					toSink = 0
				self.graph.add_term_weights(vertex_index, fromSource, toSink)

				if x > 0:
					w = self.left_weight[y, x]
					self.graph.add_edges(vertex_index, vertex_index-1, w, w)
				if x > 0 and y > 0:
					w = self.upleft_weight[y, x]
					self.graph.add_edges(vertex_index, vertex_index-self.cols-1, w, w)
				if y > 0:
					w = self.up_weight[y, x]
					self.graph.add_edges(vertex_index, vertex_index-self.cols, w, w)
				if x < self.cols - 1 and y > 0:
					w = self.upright_weight[y, x]
					self.graph.add_edges(vertex_index, vertex_index-self.cols+1, w, w)

	def estimate_segmentation(self):
		a =  self.graph.max_flow()
		for y in range(self.rows):
			for x in range(self.cols):
				if self._mask[y, x] == self._GC_PR_BGD or self._mask[y, x] == self._GC_PR_FGD:
					if self.graph.insource_segment(y*self.cols+x): # Vertex Index
						self._mask[y, x] = self._GC_PR_FGD
					else:
						self._mask[y, x] = self._GC_PR_BGD

	def iter(self, n):
		for i in range(n):
			self.assign_GMM_components()
			self.learn_GMM_parameters()
			self.construct_gcgraph(self.lam)
			self.estimate_segmentation()

	def run(self):
		self.init_with_kmeans()
		self.iter(1)

	def _smoothing(self):
		for y in range(1, self.rows-2):
			for x in range(1, self.cols-2):
				a = self._mask[x-1, y]
				b = self._mask[x+1, y]
				c = self._mask[x, y-1]
				d = self._mask[x, y+1]
				if a==b==3 or a==c==3 or a==d==3 or b==c==3 or b==d==3 or c==d==3:
					self._mask[x, y] = 3

	def show(self, output):
		FGD = np.where((self._mask == 1) + (self._mask == 3), 255, 0).astype('uint8')
		output = cv2.bitwise_and(self.img2, self.img2, mask = FGD)
		return output

if __name__ == '__main__':
	img = cv2.imread('teddy.jpg', cv2.IMREAD_COLOR)
	output = np.zeros(img.shape,np.uint8)
	GC = GCClient(img, k = 5)
	cv2.namedWindow('output')
	cv2.namedWindow('input')
	a = cv2.setMouseCallback('input',GC.init_mask)
	cv2.moveWindow('input',img.shape[0]+100, img.shape[1]+100)
	count = 0
	flag = False
	print("Draw a rectangle around the object using right mouse button \n")
	print('Press N to continue \n')
	while(1):
		cv2.imshow('output', output)
		cv2.imshow('input', np.asarray(GC.img, dtype = np.uint8))
		k = 0xFF & cv2.waitKey(1)
		if k == 27:
			break
		elif k == ord('n'):
			print(""" For finer touchups, mark foreground and background after pressing keys 0-3
			and again press 'n' \n""")
			if GC.rect_or_mask == 0:
				GC.run()
				GC.rect_or_mask = 1
			elif GC.rect_or_mask == 1:
				GC.iter(1)
			flag = True
		elif k == ord('1'):
			print('Mark background regions with left mouse button \n')
			GC._DRAW_VAL = GC._DRAW_BG
			flag = True
		elif k == ord('0'):
			print('Mark foreground regions with left mouse button \n')
			GC._DRAW_VAL = GC._DRAW_FG
			flag = True
		elif k == ord('2'):
			print('Mark prob. background regions with left mouse button \n')
			GC._DRAW_VAL = GC._DRAW_PR_BG
			flag = True
		elif k == ord('3'):
			print('Mark prob. foreground regions with left mouse button \n')
			GC._DRAW_VAL = GC._DRAW_PR_FG
			flag = True
		elif k == ord('s'):
			cv2.imwrite('%s_gc.jpg'%('hyh'), output)
			print("Result saved as image %s_gc.jpg"%('hyh'))
		elif k == ord('r'):
			GC.__init__(img, k = 5)
	
		FGD = np.where((GC._mask == 1) + (GC._mask == 3), 255, 0).astype('uint8')
		output = cv2.bitwise_and(GC.img2, GC.img2, mask = FGD)
	cv2.destroyAllWindows()
