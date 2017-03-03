#################################
# Testing Perceptron as a whole #
#################################

import perceptron as p
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

bin1 = p.Perceptron([0, 1], 2)
bin1.train(trainData1, trainLabels1)
# print("Test 1 without bias:")
result = testBatch(bin1, testData1)
if result != testLabels1:
	print 'error A1a' 
	pass

bin2 = p.Perceptron([0, 1], 3)
bin2.train(trainDataWithBias1, trainLabels1)
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

bin3 = p.Perceptron([0, 1], 9)
bin3.train(trainData2, trainLabels2)
# print("Test 2 without bias:")
result = testBatch(bin3, testData2)
# not sure what to do with this result, it should be nearish 50% though

bin4 = p.Perceptron([0, 1], 10)
bin4.train(trainDataWithBias2, trainLabels2)
# print("Test 2 with bias:")
result = testBatch(bin4, testDataWithBias2)
if result != testLabels2:
	print 'error A2'
	pass


# TEST A3 #

# multi-category classification 
# this data is a collection of (x, y) coordinates
# this data is supposed to be linearly seperable (only with a bias) by the lines x + y = 7 and x + y = 12
# i get 39% (+ or - 3%) without a bias term, 100% with one

trainData3 = [[random.randint(0, 10), random.randint(0, 10)] for _ in range(NUM_TESTS * 10)]
trainLabels3 = []
for x, y in trainData3:
	if x + y > 12:
		trainLabels3.append(2)
	elif x + y > 7:
		trainLabels3.append(1)
	else:
		trainLabels3.append(0)
trainData3 = np.array(trainData3)
trainDataWithBias3 = abttd(trainData3)

testData3 = [[random.randint(0, 10), random.randint(0, 10)] for _ in range(NUM_TESTS)]
testLabels3 = []
for x, y in testData3:
	if x + y > 12:
		testLabels3.append(2)
	elif x + y > 7:
		testLabels3.append(1)
	else:
		testLabels3.append(0)
testData3 = np.array(testData3)
testDataWithBias3 = abttd(testData3)


bin5 = p.Perceptron([0, 1, 2], 2)
bin5.train(trainData3, trainLabels3)
# print("Test 3 without bias:")
result = testBatch(bin5, testData3)
# not sure what to do with this result, it ranges alot

bin6 = p.Perceptron([0, 1, 2], 3)
bin6.train(trainDataWithBias3, trainLabels3)
# print("Test 3 with bias:")
result = testBatch(bin6, testDataWithBias3)
if result != testLabels3:
	print 'error A3'
	pass




