import numpy as np


class MaxPool:

    def forward(self, input):

        self.last_input = input

        h, w, num_filters = input.shape

        input_reshaped = input.reshape(
            h // 2,
            2,
            w // 2,
            2,
            num_filters
        )

        return input_reshaped.max(
            axis=(1, 3)
        )

    def backward(self, d_L_d_out):

        input = self.last_input

        h, w, num_filters = input.shape

        new_h = h // 2
        new_w = w // 2

        # (new_h,2,new_w,2,F)
        regions = input.reshape(
            new_h,
            2,
            new_w,
            2,
            num_filters
        )

        # mover para tener:
        # (new_h,new_w,F,2,2)
        regions = regions.transpose(
            0, 2, 4, 1, 3
        )

        # aplanar 2x2
        # (new_h,new_w,F,4)
        flat = regions.reshape(
            new_h,
            new_w,
            num_filters,
            4
        )

        # primer máximo -> mismo que v1
        max_idx = np.argmax(
            flat,
            axis=3
        )

        gradient = np.zeros_like(input)

        # convertir 0..3 a coords 2x2
        row = max_idx // 2
        col = max_idx % 2

        i_idx = (
            np.arange(new_h)[:, None, None] * 2
            + row
        )

        j_idx = (
            np.arange(new_w)[None, :, None] * 2
            + col
        )

        f_idx = np.arange(
            num_filters
        )[None, None, :]

        gradient[
            i_idx,
            j_idx,
            f_idx
        ] = d_L_d_out

        return gradient