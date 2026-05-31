import numpy as np


class ReLU:

    def forward(self, input):

        self.last_input = input

        return np.maximum(0, input)

    def backward(self, d_L_d_out):

        gradient = d_L_d_out.copy()

        gradient[self.last_input <= 0] = 0

        return gradient