import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class ConvLayer:

    def __init__(self, num_filters):

        self.num_filters = num_filters

        self.filters = np.random.randn(
            num_filters,
            3,
            3,
            3
        ).astype(np.float32) / 9

    def forward(self, input):

        self.last_input = input

        # (30,30,3,3,3)
        windows = sliding_window_view(
            input,
            (3, 3),
            axis=(0, 1)
        )

        # reorganizar canales
        windows = windows.transpose(0, 1, 3, 4, 2)

        # convolución vectorizada
        output = np.tensordot(
            windows,
            self.filters,
            axes=([2, 3, 4], [1, 2, 3])
        ).astype(np.float32)

        return output

    def backward(self, d_L_d_out, learning_rate):

        windows = sliding_window_view(
            self.last_input,
            (3, 3),
            axis=(0, 1)
        ).astype(np.float32)

        windows = windows.transpose(0, 1, 3, 4, 2)

        # gradiente filtros vectorizado
        d_L_d_filters = np.tensordot(
            d_L_d_out,
            windows,
            axes=([0, 1], [0, 1])
        ).astype(np.float32)

        self.filters -= learning_rate * d_L_d_filters

        return None