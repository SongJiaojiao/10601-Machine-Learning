import numpy as np
from logsum import log_sum
from sys import argv
from logsum import log_sum

scripts, dev, hmm_trans, hmm_emit, hmm_prior = argv

def cleanstring(string):
	a = string.split(':')
	return a[1]
def getname(string):
	a = string.split(':')
	return a[0]

def readfile(filename):
	with open(filename) as t:
		insts = []
		for line in t.readlines():
			line = line.strip()
			line = line.split(' ')
			insts.append(line)
	return np.array(insts)

trans_initial = readfile(hmm_trans)
emit_initial = readfile(hmm_emit) 
prior_initial = readfile(hmm_prior) 
sentence = readfile(dev)  # Get Observa Data
prior = np.delete(prior_initial,0,axis = 1)

def cleanmatrix(matrix):
	namelist = []
	matrix = np.delete(matrix,0,axis = 1) # Delete first column	
	for j in range(matrix.shape[1]):
		name = getname(matrix[0][j])
		namelist.append(name)
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			matrix[i][j] = cleanstring(matrix[i][j])	
	return namelist, matrix

list_states, trans = cleanmatrix(trans_initial)
list_obser, emit = cleanmatrix(emit_initial)

# Numerize
trans = trans.astype(np.float)  
emit = emit.astype(np.float)
prior = prior.astype(np.float)

# Logerize
trans_log = np.log(trans)
emit_log = np.log(emit)
prior_log = np.log(prior)
# print trans_log
# Get Emission for one Sentence
def getB(B_initial, O):
	row = B_initial.shape[0]
	col = len(O)
	B = np.zeros((row,col))
	count = 0
	for i in O:
		if i in list_obser:
			index = list_obser.index(i)  # Get column index
			B[:,count] = B_initial[:,index]
			count += 1
	return B

def backward(A,B_initial,PI,O):
	row = A.shape[0]
	col = len(O)
	Beta = np.zeros((row,col)) # Build Beta
	B = getB(B_initial,O)	
	# Initialize Beta 1
	Beta[:,(col-1)] = 0
	
	for t in reversed(range(0,len(O)-1)):
		for n in range(row):
			loopc = 0
			for i,j,k in zip(Beta[:,(t+1)],A[n,:],B[:,(t+1)]):
				if loopc == 0:
					roni = i+j+k
					loopc += 1
				else:
					roni = log_sum(roni,i+j+k)
			
			Beta[n,t] = roni
	# Logsum the first column:
	# I still don't understand why we need to logsum all of the things
	count = 0

	for i,j,k in zip(Beta[:,0],B[:,0],PI[:,0]):
		if count ==0:
			sumup = i+j+k
			count += 1
		else:
			sumup = log_sum(sumup,i+j+k)
	print sumup

for s in sentence:
	backward(trans_log,emit_log,prior_log,s)
