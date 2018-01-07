import sys
from sys import argv
import math

scripts, trainset= argv

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
	vocab = sum(vocab,[])
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

text_lib = sum(text_lib,[])      # set of lib
len_lib = len(text_lib)          # Length lib ~~~~~~~~~
text_con = sum(text_con,[])      # set of con
len_con = len(text_con)          # Length of con ~~~~~~~~~

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

unionkey = dict(train_con, **train_lib) #list


for key in train_con:
	train_con[key] = math.log(train_con[key] + float(1))- math.log(float(len_con)+len_vocab)
for key in train_lib:
	train_lib[key] = math.log(train_lib[key]+float(1)) - math.log(float(len_lib) + len_vocab)

data_lib = {}
data_con = {}
con_add = math.log(float(1)) - math.log(float(len_con) + len_vocab)
lib_add = math.log(float(1)) - math.log(float(len_lib) + len_vocab)
for key in unionkey:
	if key in train_con:
		if key in train_lib:
			data_lib[key] = train_lib[key] - train_con[key]
			data_con[key] = train_con[key] - train_lib[key]
		else:
			data_lib[key] = lib_add - train_con[key]
			data_con[key] = train_con[key] - lib_add
	else:
		data_con[key] = con_add - train_lib[key]
		data_lib[key] = train_lib[key] - con_add

consort = sorted(data_con.iteritems(), key=lambda d:d[1], reverse = True)
libsort = sorted(data_lib.iteritems(), key=lambda d:d[1], reverse = True)


#Output top20----------------------------------------

for i in range(20):
	print libsort[i][0],"%.4f" % libsort[i][1]
print ''
for i in range(20):
	print consort[i][0],"%.4f" % consort[i][1]




