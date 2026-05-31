import numpy as np


class Softmax:

    def forward(self, input):

        shifted = input - np.max(input)

        exp = np.exp(shifted)

        return exp / np.sum(exp)
    
    