import numpy as np


class MaxPool:

    def iterate_regions(self, image):

        h, w, num_filters = image.shape

        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):

                region = image[
                    i*2:i*2+2,
                    j*2:j*2+2
                ]

                yield region, i, j

    def forward(self, input):

        self.last_input = input

        h, w, num_filters = input.shape

        output = np.zeros(
            (h // 2, w // 2, num_filters)
        )

        for region, i, j in self.iterate_regions(input):

            for f in range(num_filters):

                max_value = region[0, 0, f]

                for i2 in range(2):
                    for j2 in range(2):

                        if region[i2, j2, f] > max_value:

                            max_value = region[i2, j2, f]

                output[i, j, f] = max_value

        return output

    def backward(self, d_L_d_out):

        d_L_d_input = np.zeros(
            self.last_input.shape
        )

        for region, i, j in self.iterate_regions(
            self.last_input
        ):

            _, _, num_filters = region.shape

            for f in range(num_filters):

                max_value = region[0, 0, f]

                max_i = 0
                max_j = 0

                for i2 in range(2):
                    for j2 in range(2):

                        if region[i2, j2, f] > max_value:

                            max_value = region[i2, j2, f]

                            max_i = i2
                            max_j = j2

                d_L_d_input[
                    i*2 + max_i,
                    j*2 + max_j,
                    f
                ] = d_L_d_out[i, j, f]

        return d_L_d_input