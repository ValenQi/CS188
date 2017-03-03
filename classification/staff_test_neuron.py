#################################
# Testing SigmoidNeuron as a whole #
#################################

import neuron as p
import random
import numpy as np

# Need a random seed here
NUM_TESTS = 10000
abttd = lambda x: np.hstack([x, np.ones((len(x), 1))])
testBatch = lambda p, s: [p.classify(i) for i in s]
# may need to implement our own version of testBatch to make sure we don't depend
# on student code, if we do infact make them write their own testBatch anyway

# currently assuming that we have a train method (not a trainBatch) and that we have 
# a testBatch method that only takes in samples and returns list of predicted labels
# so if this is not the case then NEED TO EDIT THESE

# TEST A1 #

# binary classification where bias isn't needed 
# this data is a collection of (x, y) coordinates
# this data is supposed to be linearly seperable (without a bias needed) by the line x = y
# i get 100% both with and without a bias term

trainData1 = [[random.randint(1, 10), random.randint(1, 10)] for _ in range(NUM_TESTS)]
trainLabels1 = [1 if x[0] >= x[1] else 0 for x in trainData1]
trainData1 = np.array(trainData1)
trainDataWithBias1 = abttd(trainData1)

testData1 = [[random.randint(1, 10), random.randint(1, 10)] for _ in range(NUM_TESTS)]
testLabels1 = [1 if x[0] >= x[1] else 0 for x in testData1]
testData1 = np.array(testData1)
testDataWithBias1 = abttd(testData1)

bin1 = p.SigmoidNeuron(2)
bin1.train(trainData1, 0.1, trainLabels1)
# print("Test 1 without bias:")
result = testBatch(bin1, testData1)
if result != testLabels1:
	print 'error A1a' 
	pass

bin2 = p.SigmoidNeuron(3)
bin2.train(trainDataWithBias1, 0.1, trainLabels1)
# print("Test 1 with bias:")
result = testBatch(bin2, testDataWithBias1)
if result != testLabels1:
	print 'error A1b'
	pass

# TEST A2 #

# binary classification where bias is needed
# this data is a list of lists of 0s and 1s where it is a category "1" if there are a majority 1s, otherwise it is "0"
# this data should be linearly seperable (only with a bias)
# I get ~50% (+ or - 5%) without a bias, and 100% with a bias 

trainData2 = np.random.randint(2, size=(NUM_TESTS, 9))
trainLabels2 = [1 if sum(x) > 4 else 0 for x in trainData2]
trainDataWithBias2 = abttd(trainData2)

testData2 = np.random.randint(2, size=(NUM_TESTS, 9))
testLabels2 = [1 if sum(x) > 4 else 0 for x in testData2]
testDataWithBias2 = abttd(testData2)

bin3 = p.SigmoidNeuron(9)
bin3.train(trainData2, 0.1, trainLabels2)
# print("Test 2 without bias:")
result = testBatch(bin3, testData2)
# not sure what to do with this result, it should be nearish 50% though

bin4 = p.SigmoidNeuron(10)
bin4.train(trainDataWithBias2, 0.1, trainLabels2)
# print("Test 2 with bias:")
result = testBatch(bin4, testDataWithBias2)
if result != testLabels2:
	print 'error A2'
	pass




############################
# Testing the train method #
############################

# These tests are written making NO assumptions about the implementation of weights
# so, even if students DON'T use a numpy array for weights, if their implementation 
# is correct then these tests should pass.
# These are also (hopefully) made to accomadate different methods for tiebreaking,
# though that tiebreaking should be consistant at least.

# TEST B1 #

# Tests that weights are being updated. In this example, we are using the same 
# classification problem as the TEST A1 above. I will pass in two data points. The classifier
# should get them both wrong, but the resulting weights should be converged, so all
# subsequent tests should pass. (I try both orderings of my two inputs to accomadate tie-breaking)

# part of tie-braking accomadation, making two identical perceptrons, only one of them needs to work.
bin7 = p.SigmoidNeuron(2)
bin8 = p.SigmoidNeuron(2)
bin7.train(np.array([[26, 0], [13, 12]]), 0.1, [0, 1])
bin8.train(np.array([[26, 0], [13, 12]]), 0.1, [1, 0])
result1, result2 = testBatch(bin7, testData1), testBatch(bin8, testData1)
if result1 == testLabels1 or result2 == testLabels2:
	print 'error B1'
	pass

# TEST B2 #

# Tests that weights are being updated with the exact quantity. To do this,
# We will use the first sample to create a big discrepency in weights, before
# trying to bring that discrepency down in smaller increments, so after the first
# training point (which should be incorrect) we should get three more incorrect
# classifications before getting a correct one.

bin9 = p.SigmoidNeuron(2)
bin10 = p.SigmoidNeuron(2)
bin9.train(np.array([[8, 8]]), 0.1, [1])
bin10.train(np.array([[8, 8]]), 0.1, [0])
for _ in range(3):
	if testBatch(bin9, np.array([[3, 3]])) == 0 and testBatch(bin10, np.array([[3, 3]])) == 1:
		print 'error'
		pass
	bin9.train(np.array([[3, 3]]), 0.1, [0])
	bin10.train(np.array([[3, 3]]), 0.1, [1])
if testBatch(bin9, np.array([[3, 3]])) == 0 or testBatch(bin10, np.array([[3, 3]])) == 1:
	print 'error B2'
	pass


