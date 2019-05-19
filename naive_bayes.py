import numpy as np
from PIL import Image
import math
import sys

arguments = sys.argv[1:]
count = len(arguments)

if (count != 2):
	exit()


path_train_file = arguments[0]
path_test_file = arguments[1]

def calcprob_normal(X, mean, covar, p_c):
	p = -0.5*math.log(np.linalg.det(covar)) -0.5*(np.matmul(np.matmul((X-mean).T , np.linalg.inv(covar)) , (X-mean))) + math.log(p_c)
	return p

def diagonalize(M):
	d = np.diagonal(M)
	M_r = np.zeros(M.shape)
	cnt = 0
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			if i==j:
				M_r[i][j] = d[cnt]
				cnt=cnt+1
	return M_r


def dataset_loader_train(filename, dim=256):
	A = []
	frr = np.loadtxt(filename, delimiter=" ", dtype=str)
	names = frr[:, 0]
	labels = frr[:, 1]
	unique_labels = np.unique(labels)
	num_classes = unique_labels.shape[0]
	for i in range(names.shape[0]):
		img = image_imread(names[i], dim, dim)
		img = img.reshape(img.shape[0]*img.shape[1])
		A.append(img)
	A = np.asarray(A)
	return A, unique_labels, labels


def estimate_mean_varience(images, unique_labels, labels):
	num_classes = unique_labels.shape[0]
	mean = []
	var = []
	classprob = []
	for i in range(unique_labels.shape[0]):
		B = []		
		allindices = np.argwhere(labels == unique_labels[i])
		classprob.append(float(allindices.shape[0])/float(labels.shape[0]))
		for j in range(allindices.shape[0]):
			B.append(images[allindices[j]].flatten())
		B = np.asarray(B)
		m = np.mean(B, axis=0)
		v = np.cov(B.T)
		v = diagonalize(v)
		var.append(v)
		mean.append(m)
	mean = np.asarray(mean)
	var = np.asarray(var)
	classprob = np.asarray(classprob)
	return mean, var, classprob

def dataset_loader_test(filename, dim=256):
	A = []
	frr = np.loadtxt(filename, dtype=str)
	names = frr
	for i in range(names.shape[0]):
		img = image_imread(names[i], dim, dim)
		img = img.reshape(img.shape[0]*img.shape[1])
		A.append(img)
	A = np.asarray(A)
	return A

def project_data(arr, vec, N=32):
	feature_vectors = []
	for i in range(arr.shape[0]):
		I = arr[i]
		I = I.reshape((I.shape[0]), 1)
		pt = np.matmul(I.T, vec[:, 0:N])
		pt=pt.reshape(pt.shape[1])
		feature_vectors.append(pt)
	feature_vectors = np.asarray(feature_vectors)
	return feature_vectors
	
def find_pca(A):
	Mean = np.mean(A, axis=1)
	Mean= Mean.reshape(Mean.shape[0],1)
	Var = A - Mean
	U, S, V = np.linalg.svd(Var, full_matrices=False)
	return S, U, Mean, Var


def image_imread(img_name, dim1=0, dim2=0):
	I = Image.open(img_name)
	I = I.convert("L")
	I = np.asarray(I)
	if (dim1!=0 and dim2!=0):
		I = np.resize(I, (dim1,dim2))

	return I

def image_resize(I, dim):
	I = np.resize(I, (dim,dim))
	return I

A, unique_labels, lbs = dataset_loader_train(path_train_file)
labels = lbs
I = A.copy()
A = A.T
val, vec, M, ma_data = find_pca(A)
feature_vectors = project_data(I, vec)
allmean, allvar, clsprob = estimate_mean_varience(feature_vectors, unique_labels, lbs)
testimages = dataset_loader_test(path_test_file)
test_vectors = project_data(testimages, vec)
for i in range(test_vectors.shape[0]):
	maxprob = -1000000000
	for j in range(allmean.shape[0]):
		prob = calcprob_normal(test_vectors[i], allmean[j], allvar[j], clsprob[j])
		if prob > maxprob:
			maxprob=prob
			maxc = j
	print (unique_labels[maxc])











