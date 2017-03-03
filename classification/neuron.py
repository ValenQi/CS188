import numpy as np 
import samples
import random
import data_classification_utils as dcu
import neuron_utils as nu
import matplotlib.pyplot as plt
import math
from util import raiseNotDefined


class SigmoidNeuron(object):
    def __init__(self, numFeatures):
        """numFeatures: Number of features. This Neuron is a binary classifier
           and thus does not need a list of categories."""
        self.numFeatures = numFeatures

        """YOUR CODE HERE"""
        raiseNotDefined()

    def _sigmoid(self, z):
      """Function that maps dot product values to activation level. -infinity
      is mapped to 0, and infinity is mapped to 1. Can be interpreted as the
      probability that the sample is in the positive class.

      This may look different than softmax, but don't be fooled, it's actually
      the same if you divide top and bottom by e^z."""
      return 1.0 / (1 + math.exp(-2*z))

    def loss(self, sample, label, w):
        """sample: np.array of shape(1, numFeatures).
        label:  the correct label of the sample 
        w:      the weight vector under which to calculate loss

        Can interpret loss as the probability of being in the correct class. 
        For numerical accuracy reasons, the loss is expressed as 
        math.log(1/sigmoid) instead of -math.log(sigmoid) as we discussed in class.

        Note: This function uses the weight passed in, not the weight vector
        of this neruon! Do not modify this function."""
        z = np.dot(w, sample)

        if label == True:
            return math.log(1.0 + math.exp(-2*z))
        else:
            return math.log(1.0 + math.exp(2*z))

    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: True if activation is greater than 0.5."""

        """YOUR CODE HERE"""
        raiseNotDefined()
    
    def train_single(self, sample, alpha, label):
        """Performs stochastic gradient descent on a single sample, i.e.
        subtracts alpha * gradient form the current set of weights.
         
        Hint: You'll want to use nu.gradient to find the gradient
        of the loss function for this data point."""
        
        """YOUR CODE HERE"""
        raiseNotDefined()

    def train(self, samples, alpha, labels):
        """Performs one iteration of stochastic gradient descent 
        by training over the entire set of samples (one sample at a time)."""

        """YOUR CODE HERE"""
        raiseNotDefined()

