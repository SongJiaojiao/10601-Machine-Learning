import sys
from sys import argv
import math

scripts, trainset,testset = argv

# Read single file,return in list-------------------
def loadfile(filename):
	insts = []
	with open(filename,"r") as f:
		for line in f:
			line = line.strip() # Remove blank
			line = line.lower() # Transform lowercase
			insts.append(line)
	return insts

filelist = loadfile(trainset)
testlist = loadfile(testset)

# Calculate vocab  --------------------------------
def GetVocab(filename):
	vocab = []
	for i in filelist:
		data = loadfile(i)
		vocab.append(data)
	vocab     = sum(vocab,[])
	diffvocab = set(vocab)
	len_vocab = len(diffvocab) #|vocab| ~~~~~~~~~
	return diffvocab, len_vocab

all_vocab_words,len_vocab = GetVocab(trainset)


# Calculate Prior--------------
libfile = 0
confile = 0
text_lib = []
text_con = []
for i in filelist:
	a = loadfile(i)
	if i.startswith('lib'):
		text_lib.append(a)
		libfile += 1
	elif i.startswith('con'):
		text_con.append(a)
		confile += 1
priorlib = math.log(float(libfile)/(libfile+confile))  # Prior(log) of lib ~~~~~~~~~
priorcon = math.log(float(confile)/(libfile+confile))  # Prior(log) of con ~~~~~~~~~

# Get files for each label-------
text_lib = sum(text_lib,[])      # Aggregate set of lib
text_con = sum(text_con,[])      # Aggregate set of con


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


Nlib = 0.0
Ncon = 0.0
for key in train_lib:
	Nlib += train_lib[key]
for key in train_con:
	Ncon += train_con[key]



def predict(testset):
	u = 1.0
	templib = 0.0
	tempcon = 0.0
	MAPlib = 0.0
	MAPcon = 0.0
	testdata = loadfile(testset)
	for i in testdata:
		if i in all_vocab_words:
			try:
				templib = math.log(train_lib[i]+float(u)) - math.log (float(Nlib) + len_vocab)
			except:
				templib = math.log(float(u)) - math.log (float(Nlib) + len_vocab)
			MAPlib += templib
			try:
				tempcon = math.log(train_con[i]+float(u)) - math.log(float(Ncon) + len_vocab)
			except:
				tempcon = math.log(float(u)) - math.log(float(Ncon) + len_vocab)				
			MAPcon += tempcon		

	MAPlib += priorlib
	MAPcon += priorcon
	if MAPlib > MAPcon:
		return 'L'
	else:
		return 'C'

#test-------------
correct = []
result = []
for i in testlist: 
	if i.startswith("lib"):
		correct.append('L')
	elif i.startswith("con"):
		correct.append('C')
	result.append(predict(i))

for i in result:
	print i
	
correctnum = 0
total = len(correct)

for a,b in zip(correct,result):
	if a == b:
		correctnum+=1
print "Accuracy:","%.4f" %(float(correctnum)/total)





