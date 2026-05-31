import numpy as np


class Softmax:

    def forward(self, input):

        max_value = input[0]

        for i in range(len(input)):

            if input[i] > max_value:
                max_value = input[i]

        exp_values = np.zeros(len(input))

        total = 0.0

        for i in range(len(input)):

            exp_values[i] = np.exp(
                input[i] - max_value
            )

            total += exp_values[i]

        output = np.zeros(len(input))

        for i in range(len(input)):

            output[i] = (
                exp_values[i] / total
            )

        return output