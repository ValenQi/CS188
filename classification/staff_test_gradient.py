import numpy as np
import neuron_utils

def g(w):
    w1 = w[0]
    w2 = w[1]
    return w1 ** 3 * w2 + 3 * w1

nabla = neuron_utils.gradient(g, np.array([3, 4], dtype='f'))
if nabla[0] < 110 or nabla[0] > 112:
    print("First component of gradient should be between 110 and 112")
if nabla[1] < 26 or nabla[1] > 28:
    print("Second component of gradient should be between 26 and 28")