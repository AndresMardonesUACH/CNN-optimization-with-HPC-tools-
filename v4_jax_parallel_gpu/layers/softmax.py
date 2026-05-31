import jax.numpy as jnp


class Softmax:

    def forward(
        self,
        input
    ):

        shifted = (
            input
            -
            jnp.max(input)
        )

        exp_values = jnp.exp(
            shifted
        )

        return (
            exp_values
            /
            jnp.sum(exp_values)
        )