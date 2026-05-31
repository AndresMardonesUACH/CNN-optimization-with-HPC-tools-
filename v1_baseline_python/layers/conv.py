import numpy as np


class ConvLayer:

    def __init__(self, num_filters):

        self.num_filters = num_filters

        self.filters = np.random.randn(
            num_filters,
            3,
            3,
            3
        ) / 9

    def iterate_regions(self, image):

        h, w, _ = image.shape

        for i in range(h - 2):
            for j in range(w - 2):

                region = image[
                    i:i+3,
                    j:j+3
                ]

                yield region, i, j

    def forward(self, input):

        self.last_input = input

        h, w, _ = input.shape

        output = np.zeros(
            (h - 2, w - 2, self.num_filters)
        )

        for region, i, j in self.iterate_regions(input):

            for f in range(self.num_filters):

                total = 0.0

                for i2 in range(3):
                    for j2 in range(3):
                        for c in range(3):

                            total += (
                                region[i2, j2, c]
                                *
                                self.filters[f, i2, j2, c]
                            )

                output[i, j, f] = total

        return output

    def backward(self, d_L_d_out, learning_rate):

        d_L_d_filters = np.zeros(
            self.filters.shape
        )

        for region, i, j in self.iterate_regions(
            self.last_input
        ):

            for f in range(self.num_filters):

                for i2 in range(3):
                    for j2 in range(3):
                        for c in range(3):

                            d_L_d_filters[
                                f,
                                i2,
                                j2,
                                c
                            ] += (
                                d_L_d_out[i, j, f]
                                *
                                region[i2, j2, c]
                            )

        for f in range(self.num_filters):
            for i in range(3):
                for j in range(3):
                    for c in range(3):

                        self.filters[f, i, j, c] -= (
                            learning_rate
                            *
                            d_L_d_filters[f, i, j, c]
                        )

        return None