import jax.numpy as jnp


class MaxPool:

    def forward(
        self,
        input
    ):

        self.last_input = input

        h, w, num_filters = input.shape

        x = input.reshape(
            h // 2,
            2,
            w // 2,
            2,
            num_filters
        )

        output = jnp.max(
            x,
            axis=(1, 3)
        )

        return output

    def backward(
        self,
        d_L_d_out
    ):

        x = self.last_input

        h, w, num_filters = x.shape

        new_h = h // 2
        new_w = w // 2

        # (new_h,2,new_w,2,F)
        regions = x.reshape(
            new_h,
            2,
            new_w,
            2,
            num_filters
        )

        # (new_h,new_w,F,2,2)
        regions = regions.transpose(
            0,
            2,
            4,
            1,
            3
        )

        # (new_h,new_w,F,4)
        flat = regions.reshape(
            new_h,
            new_w,
            num_filters,
            4
        )

        # primer máximo
        max_idx = jnp.argmax(
            flat,
            axis=3
        )

        row = max_idx // 2
        col = max_idx % 2

        gradient = jnp.zeros_like(x)

        i_idx = (
            jnp.arange(new_h)[:, None, None] * 2
            + row
        )

        j_idx = (
            jnp.arange(new_w)[None, :, None] * 2
            + col
        )

        f_idx = jnp.arange(
            num_filters
        )[None, None, :]

        gradient = gradient.at[
            i_idx,
            j_idx,
            f_idx
        ].set(
            d_L_d_out
        )

        return gradient