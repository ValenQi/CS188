import numpy as np
import math
import util
import matplotlib.pyplot as plt
from util import raiseNotDefined


def display_digit_features(weights, bias):
    """Visualizes a set of weight vectors for each digit. Assumes that you are passing
       in a numpy array with 10 columns and numFeatures rows. Depending on how you
       implemented your Perceptron weights, you might need to write another helper
       function to get your weights into this format. 

       Do not modify this code."""
    feature_matrices = []
    for i in range(10):
      feature_matrices.append(convert_weight_vector_to_matrix(weights[:, i], 28, 28, bias))

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(feature_matrices[i], cmap='gray')
        
    plt.show()

def apply_bias(samples):
    """
    samples: The samples under test, should be a numpy array of shape (numSamples, numFeatures).
    Mutates samples to add a bias term to each feature vector. This example appends a 1 to
    the front, but the bias term could be added anywhere.

    Do not modify this code."""
    return np.hstack([samples, np.ones((len(samples), 1))])

def simple_image_featurization(image):
    """Converts an image to a numpy vector of shape (1, w * h), where w is the
        width of the image, and h is the height."""

    """YOUR CODE HERE"""
    raiseNotDefined()

def zero_one_loss_ss(classifier, sample, label):
    """
    classifier: The classifier under test.
    sample: The sample under test, should be a numpy array of shape (1, numFeatures).
    label: The correct label of the sample under test.

    Returns 0.0 if the classifier classifies the sample correctly, or 1.0 otherwise."""

    """YOUR CODE HERE"""
    raiseNotDefined()


def zero_one_loss(classifier, samples, labels):
    """
    classifier: The classifier under test.
    sample: The samples under test, should be a numpy array of shape (numSamples, numFeatures).
    label: The correct labels of the samples under test.

    Returns the fraction of samples that are wrong. For example, if the classifier gets
    65 out of 100 samples right, this function should return 0.35."""

    """YOUR CODE HERE"""
    raiseNotDefined()

def convert_weight_vector_to_matrix(weight_vector, w, h, bias):
    """weight_vector: The weight vector to transform into a matrix.
    w: the width of the matrix
    h: the height of the matrix
    bias: whether or not there is a bias feature

    Returns a w x h array where the first w entries of the weight vector for this label correspond to the
    first row, the next w the next row, and so forth. Assume that w * h is equal to the size of the
    weight vector. Ignore the bias if there is one"""

    """YOUR CODE HERE"""
    raiseNotDefined()


def convert_perceptron_weights_to_2D_array_with_ten_columns(p):
    """p: A Perceptron.

    Returns a numpy array with 10 columns and 784 rows if bias is false,
    and 785 rows if bias is true. 

    This function is intended to convert your Perceptron's internal 
    representation of its weight vector into the format required by 
    display_digit_features.

    If your Perceptron already stores weights in this exact format, then this
    function is not necessary. If necessary, this function could be as simple as 
    transposing your internal representation, or it could be as complicated as
    having to iterate through some kind of dictionary.

    Example output shown below:
    array([[ 0.10528018,  0.16808003, ..., 0.18949908],
           [ 0.67620099,  0.12085823, ..., 0.49560261],
           [ 0.1710934 ,  0.33713286, ..., 0.32837192],
           [ 0.34823874,  0.91873123, ..., 0.11187318],
           [ 0.88123712,  0.0012387 , ..., 0.30981232],
           [ 0.23948234,  0.01283712, ..., 0.19308969],
           [ 0.12837172,  0.85762893, ..., 0.49827312],
           [ 0.97893816,  0.02660188, ..., 0.16953317],
           [ 0.99322359,  0.89868266, ..., 0.69822413]])
    """
    raiseNotDefined()