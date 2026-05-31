import jax.numpy as jnp


class ReLU:

    def forward(
        self,
        input
    ):

        self.last_input = input

        return jnp.maximum(
            input,
            0
        )

    def backward(
        self,
        d_L_d_out
    ):

        mask = (
            self.last_input > 0
        )

        return (
            d_L_d_out
            *
            mask
        )