import numpy as np
import random

def get_size(A):
	return list(A.shape)[:2]

class kmeans:
	def __init__(self, A, dim = 2, n = 2, max_iter = 10):
		self.shape = list(A.shape)
		self.rows = int(A.size/dim)
		self.A = A.reshape(self.rows, dim)
		self.n = n
		self.types = np.zeros(self.rows, dtype = np.uint)
		self._init_centers()
		self.max_iter = max_iter

	def _init_centers(self):
		center_indexs = []
		print(self.n)
		for i in range(5):
			while True:
				rindex = random.randint(0, self.rows-1)
				if rindex not in center_indexs:
					center_indexs.append(rindex)
					break
		self.centers = self.A[center_indexs]

	def determine_types(self):
		self.types = np.asarray([np.asarray([((self.A[i] - self.centers[j]) ** 2).sum() for j in range(5)]).argmin(0) for i in range(self.rows)])

	def refresh_centers(self):
		cluster_length = []
		for i in range(5):
			index = np.where(self.types == i)
			length = len(list(index)[0])
			cluster_length.append(length)
		cluster_length = np.asarray(cluster_length)
		for i in range(5):
			if cluster_length[i] == 0:
				k = cluster_length.argmax(0)
				p = np.where(self.types == k)
				pixels = self.A[p]
				index = np.asarray([((r - self.centers[k])**2).sum() for r in pixels]).argmax(0)
				index = list(p)[0][index]
				self.types[index] = i
				self.centers[i] = self.A[index]
			else:
				index = np.where(self.types == i)
				self.centers[i] = self.A[index].sum(axis=0)/len(list(index)[0])

	def run(self):
		for i in range(self.max_iter):
			self.determine_types()
			self.refresh_centers()

	def plot(self):
		data_x = []
		data_y = []
		data_z = []
		for i in range(5):
			index = np.where(self.types == i)
			data_x.extend(self.A[index][:,0].tolist())
			data_y.extend(self.A[index][:,1].tolist())
			data_z.extend([i/5 for j in range(len(list(index)[0]))])
		sc = plt.scatter(data_x, data_y, c=data_z, vmin=0, vmax=1, s=35, alpha=0.8)
		plt.colorbar(sc)
		plt.show()

	def output(self):
		self.data_by_comp = []
		for ci in range(5):
			index = np.where(self.types == ci)
			self.data_by_comp.append(self.A[index])
		return self.data_by_comp

if __name__ == '__main__':
	A = np.random.random([1000, 3, 2])
	k = kmeans(A, n = 20, max_iter = 10)
	k.run()
	k.plot()
	r = k.output()
