import numpy as np
from PIL import Image
import math
import sys

#Arguments
arguments = sys.argv[1:]
count = len(arguments)

if (count != 2):
	exit()

path_train_file = arguments[0]
path_test_file = arguments[1]
N = 32

def dataset_loader_train(filename, dim=256):
	A = []
	frr = np.loadtxt(filename, delimiter=" ", dtype=str)
	names = frr[:, 0]
	labels = frr[:, 1]
	unique_labels = np.unique(labels)
	num_classes = unique_labels.shape[0]
	Y = []
	for i in range(names.shape[0]):
		img = image_imread(names[i], dim, dim)
		img = img.reshape(img.shape[0]*img.shape[1])
		lb = np.where(unique_labels == labels[i])
		onehot = np.zeros([num_classes])
		onehot[lb] = 1
		Y.append(onehot)
		A.append(img)
	A = np.asarray(A)
	Y = np.asarray(Y)
	return A, Y, unique_labels

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
		pt = np.matmul(I.T , vec[:, 0:N])
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


def softmax(z):
	z_norm=np.exp(z-np.max(z,axis=0,keepdims=True))
	return(np.divide(z_norm,np.sum(z_norm,axis=0,keepdims=True)))

def crossEntropy(Y, Y_p):
	s = 0
	for i in range(Y.shape[0]):
		s += (Y[i]*math.log(Y_p[i]+1e-15))
	return -s

def totalLoss(Yvec, Y_pvec):
	l = 0
	for i in range(Yvec.shape[0]):
		l = l + crossEntropy(Yvec[i], Y_pvec[i])
	return l
	
def gradient(X, Y, Y_p):
	g = np.matmul(X, (Y_p - Y))
	return g

def findLabel(Y_p):
	labels = []
	for i in range(Y_p.shape[0]):
		ohv = Y_p[i]
		labels.append(np.argmax(ohv))
	return labels

def calcAccuracy(orig, pred):
	s = 0
	for i in range(len(orig)):
		if orig[i] == pred[i]:
			s = s+1
	return s/len(orig)

def predictLabel(pred, unique_labels):
	save = []
	for i in range(pred.shape[0]):
		p = np.argmax(pred[i])
		save.append([unique_labels[p]])
	save = np.asarray(save)
	return save



A, labels, unique_labels = dataset_loader_train(path_train_file)
I = A.copy()
A = A.T
val, vec, M, ma_data = find_pca(A)
feature_vectors = project_data(I, vec, N)
feature_vectors = feature_vectors.T
orig = findLabel(labels.copy())

W = np.random.rand(N+1, unique_labels.shape[0])
X = feature_vectors.copy()
X = np.append(X, np.ones([1, X.shape[1]]), axis = 0)
epochs = 0
max_epochs = 10000
best_W = W
max_training_accuracy = 0
while(1):
	Y_p = softmax(np.matmul(W.T, X)).T
	l = totalLoss(labels.copy(), Y_p.copy())
	lab = findLabel(Y_p.copy())
	acc = calcAccuracy(orig, lab)
	if acc>max_training_accuracy:
		max_training_accuracy = acc
		best_W = W
	if(acc>0.8):
		break
	W = W - 0.0001*gradient(X.copy(), labels.copy(), Y_p.copy())
	epochs = epochs+1
	if (epochs > max_epochs):
		break

testimages = dataset_loader_test(path_test_file)
test_vectors = project_data(testimages, vec, N)
test_vectors = test_vectors.T
test_vectors = np.append(test_vectors, np.ones([1, test_vectors.shape[1]]), axis = 0)
predictions = softmax(np.matmul(best_W.copy().T, test_vectors.copy())).T
sv = predictLabel(predictions, unique_labels)
for i in range(sv.shape[0]):
	print (sv[i][0])













