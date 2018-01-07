#!/usr/bin/env python
#coding: utf-8
import sys
import csv
import math
import numpy as np
import random
import time

train_insts_filename, train_label_filename, test_insts_filename = sys.argv[1], sys.argv[2], sys.argv[3]

def load(filename):
	# Loading instances.
	insts = []
	with file(filename, "r") as fin:
		fin.readline()
		for line in fin:
			line = line.strip()
			segments = line.split(",")
			inst = [float(segments[0]), float(segments[1])]
			if segments[2] == "yes":
				inst.append(1.0)
			else:
				inst.append(0.0)
			if segments[3] == "yes":
				inst.append(1.0)
			else:
				inst.append(0.0)
			insts.append(inst)
	return np.asarray(insts, dtype=np.float)

def load_label(filename):
	# Loading labels.
	labels = []
	with file(filename, "r") as fin:
		for line in fin:
			line = line.strip()
			if line == "yes":
				labels.append(1.0)
			else:
				labels.append(0.0)
	return np.asarray(labels)

train_insts = load(train_insts_filename)
test_insts = load(test_insts_filename)
train_labels = load_label(train_label_filename)[:, np.newaxis]
test_labels = np.asarray([1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,0,0,0,0,1,1], dtype=np.float)[:, np.newaxis]

num_train = train_insts.shape[0]
num_test = test_insts.shape[0]
num_feats = train_insts.shape[1]
# Normalization.
feat_mean = np.mean(train_insts, axis=0)
feat_std = np.std(train_insts, axis=0)

train_insts -= feat_mean
train_insts /= feat_std
test_insts -= feat_mean
test_insts /= feat_std
# Augmentation.
trains = np.ones((num_train, num_feats + 1), dtype=np.float)
tests = np.ones((num_test, num_feats + 1), dtype=np.float)
trains[:, :num_feats] = train_insts
tests[:, :num_feats] = test_insts
train_insts, test_insts = trains, tests
num_feats += 1

# Building neural network.
np.random.seed(42)
lr = 0.1
num_hiddens = 50
num_iters = 5000

hiddenw = np.random.rand(num_feats, num_hiddens)
classw = np.random.rand(num_hiddens, 1)
classb = 0.0
dhiddenw = np.zeros((num_feats, num_hiddens))
dclassw = np.zeros((num_hiddens, 1))
dclassb = 0.0

def sigmoid(vec):
	return 1.0 / (1.0 + np.exp(-vec))

# Forward propagation.
def forward(inputs):
	h1 = sigmoid(np.dot(inputs, hiddenw))
	h2 = np.dot(h1, classw) + classb
	return (h1, h2), sigmoid(h2)

# Backward propagation
def backward(probs, labels, (x, h1, h2)):
    n = probs.shape[0]
    e2 = probs - labels
    e1 = np.dot(e2, classw.T)
    e1 = e1 * h1 * (1.0 - h1)
    dclassw[:] = np.dot(h1.T, e2) / n
    dclassb = np.mean(e2, axis=0)
    dhiddenw[:] = np.dot(x.T, e1) / n

start_time = time.time()
for i in xrange(num_iters):
    (h1, h2), probs = forward(train_insts)
    backward(probs, train_labels, (train_insts, h1, h2))
    # Gradient update.
    hiddenw -= lr * dhiddenw
    classw -= lr * dclassw
    classb -= lr * dclassb
    # Compute the error squared.
    squared_err = np.sum(np.square(probs - train_labels))
    train_preds = probs >= 0.5
    # print "predicted label", probs
    train_acc = np.mean(train_preds == train_labels)
    print squared_err
print "TRAINING COMPLETED! NOW PREDICTING."
end_time = time.time()
# Compute test set accuracy.
_, probs = forward(test_insts)
test_preds = probs >= 0.5
test_acc = np.mean(test_preds == test_labels)
# print "Test set accuracy", test_acc
for t in test_preds:
	if t: print "yes"
	else: print "no"
# print "Total time used for training: {} seconds".format(end_time - start_time)
