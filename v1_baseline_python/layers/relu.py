import numpy as np


class ReLU:

    def forward(self, input):

        self.last_input = input

        h, w, num_filters = input.shape

        output = np.zeros(input.shape)

        for i in range(h):
            for j in range(w):
                for f in range(num_filters):

                    if input[i, j, f] > 0:
                        output[i, j, f] = input[i, j, f]
                    else:
                        output[i, j, f] = 0

        return output

    def backward(self, d_L_d_out):

        h, w, num_filters = d_L_d_out.shape

        gradient = np.zeros(
            d_L_d_out.shape
        )

        for i in range(h):
            for j in range(w):
                for f in range(num_filters):

                    if self.last_input[i, j, f] > 0:

                        gradient[i, j, f] = (
                            d_L_d_out[i, j, f]
                        )
                    else:
                        gradient[i, j, f] = 0

        return gradient