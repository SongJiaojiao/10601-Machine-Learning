import sys
from sys import argv
import math

scripts, trainset = argv

#read file----------------------------------------
def loadfile(filename):
	insts = []
	with open(filename,"r") as f:
		for line in f:
			line = line.strip() # Remove blank
			line = line.lower() # Transform lowercase
			insts.append(line)
	return insts

filelist = loadfile(trainset)


#calculate vocab  --------------------------------
def GetVocab(filename):
	vocab = []
	for i in filelist:
		data = loadfile(i)
		vocab.append(data)
# print 'vocab:',len(vocab)
	vocab = sum(vocab,[])
# print len(vocab)
	diffvocab = set(vocab)
	len_vocab = len(diffvocab) #|vocab| ~~~~~~~~~
	return len(vocab), len_vocab

total_vocab,len_vocab = GetVocab(trainset)


#calculate sum of each label && prior--------------
lib = 0
con = 0
text_lib = []
text_con = []

for i in filelist:
	a = loadfile(i)
	if i.startswith('lib'):
		text_lib.append(a)
		lib += 1
	elif i.startswith('con'):
		text_con.append(a)
		con += 1
# print 'libfile',len(text_lib)
# print 'confile',len(text_con)
text_lib = sum(text_lib,[])      # Aggregate set of lib
len_lib = len(text_lib)          # Length lib ~~~~~~~~~
# print sum_lib
text_con = sum(text_con,[])      # Aggregate set of con
len_con = len(text_con)          # Length of con ~~~~~~~~~
# print sum_con
priorlib = math.log(float(lib)/(lib+con))  # Prior of lib ~~~~~~~~~
priorcon = math.log(float(con)/(lib+con))  # Prior of con ~~~~~~~~~
# def conditional(number):
def train(conset,libset):
	lib = {}
	con = {}
	for i in conset:
		if i in con:
			con[i] += 1			
		else:
			con[i] = 1
			
	for i in libset:
		if i in lib:
			lib[i] += 1
		else:
			lib[i] = 1
	return con,lib


train_con, train_lib = train(text_con,text_lib)
consort = sorted(train_con.iteritems(), key=lambda d:d[1], reverse = True)
libsort = sorted(train_lib.iteritems(), key=lambda d:d[1], reverse = True)


#Output top20----------------------------------------
u = 1
for i in range(20):
	libprob = (float(libsort[i][1])+u)/(len_lib+len_vocab)
	print libsort[i][0],"%.4f" % libprob
print ''
for i in range(20):
	conprob = (float(consort[i][1])+u)/(len_con+len_vocab)
	print consort[i][0],"%.4f" % conprob






