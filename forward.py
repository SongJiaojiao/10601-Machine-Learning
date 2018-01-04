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

def forward(A,B_initial,PI,O):
	row = A.shape[0]
	col = len(O)
	Alpha = np.zeros((row,col)) # Build Alpha
	B = getB(B_initial,O)	
	# Initialize Alpha 1
	Alpha[:,0] = B[:,0] + PI[:,0]

	# print Alpha[:,0]
	for t in range(1,len(O)):
		for n in range(row):
			loopc = 0
			for i,j in zip(Alpha[:,t-1],A[:,n]):
				if loopc == 0:
					roni = i+j
					loopc += 1
				else:
					roni = log_sum(roni,i+j)
			
			Alpha[n,t] = roni + B[n,t]
	for i in Alpha:
		print i


	# Logsum the last column:
	# I still don't understand why we need to logsum all of the things
	count = 0
	for i in Alpha[:,-1]:
		if count ==0:
			sumup = i
			count += 1
		else:
			sumup = log_sum(sumup,i)
	return sumup
print forward(trans_log,emit_log,prior_log,sentence[-1])
# for s in sentence:
# 	print forward(trans_log,emit_log,prior_log,s)
