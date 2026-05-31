import numpy as np


class Dense:

    def __init__(self, input_len, nodes):

        self.weights = np.random.randn(
            input_len,
            nodes
        ).astype(np.float32) / input_len

        self.biases = np.zeros(nodes).astype(np.float32)

    def forward(self, input):

        self.last_input_shape = input.shape

        input = input.flatten()

        self.last_input = input

        return input @ self.weights + self.biases

    def backward(self, d_L_d_out, learning_rate):

        # gradientes vectorizados
        d_L_d_w = np.outer(
            self.last_input,
            d_L_d_out
        ).astype(np.float32)

        d_L_d_b = d_L_d_out

        d_L_d_inputs = self.weights @ d_L_d_out

        # update
        self.weights -= learning_rate * d_L_d_w
        self.biases -= learning_rate * d_L_d_b

        return d_L_d_inputs.reshape(
            self.last_input_shape
        )