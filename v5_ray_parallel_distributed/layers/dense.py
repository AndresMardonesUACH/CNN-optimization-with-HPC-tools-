import numpy as np


class Dense:

    def __init__(self, input_len, nodes):

        self.weights = np.random.randn(
            input_len,
            nodes
        ) / input_len

        self.biases = np.zeros(nodes)

    def forward(self, input):

        self.last_input_shape = input.shape

        input_flat = input.flatten()

        self.last_input = input_flat

        output = np.zeros(
            self.biases.shape
        )

        for j in range(len(self.biases)):

            total = 0.0

            for i in range(len(input_flat)):

                total += (
                    input_flat[i]
                    *
                    self.weights[i, j]
                )

            total += self.biases[j]

            output[j] = total

        return output

    def backward(self, d_L_d_out, learning_rate):

        d_L_d_inputs = np.zeros(
            self.last_input.shape
        )

        d_L_d_weights = np.zeros(
            self.weights.shape
        )

        d_L_d_biases = np.zeros(
            self.biases.shape
        )

        for j in range(len(d_L_d_out)):

            d_L_d_biases[j] = d_L_d_out[j]

            for i in range(len(self.last_input)):

                d_L_d_weights[i, j] = (
                    self.last_input[i]
                    *
                    d_L_d_out[j]
                )

                d_L_d_inputs[i] += (
                    self.weights[i, j]
                    *
                    d_L_d_out[j]
                )

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):

                self.weights[i, j] -= (
                    learning_rate
                    *
                    d_L_d_weights[i, j]
                )

        for j in range(len(self.biases)):

            self.biases[j] -= (
                learning_rate
                *
                d_L_d_biases[j]
            )

        return d_L_d_inputs.reshape(
            self.last_input_shape
        )