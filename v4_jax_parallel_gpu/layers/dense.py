import jax
import jax.numpy as jnp


class Dense:

    def __init__(
        self,
        input_len,
        nodes
    ):

        key = jax.random.PRNGKey(1)

        self.weights = (
            jax.random.normal(
                key,
                (
                    input_len,
                    nodes
                )
            )
            / input_len
        ).astype(jnp.float32)

        self.biases = jnp.zeros(
            nodes,
            dtype=jnp.float32
        )

    def forward(
        self,
        input
    ):

        self.last_input_shape = (
            input.shape
        )

        input_flat = input.reshape(-1)

        self.last_input = input_flat

        output = (
            jnp.dot(
                input_flat,
                self.weights
            )
            +
            self.biases
        )

        return output

    def backward(
        self,
        d_L_d_out,
        learning_rate
    ):

        # grad input
        d_L_d_inputs = jnp.dot(
            self.weights,
            d_L_d_out
        )

        # grad weights
        d_L_d_weights = jnp.outer(
            self.last_input,
            d_L_d_out
        )

        # grad bias
        d_L_d_biases = d_L_d_out

        # update
        self.weights = (
            self.weights
            -
            learning_rate
            *
            d_L_d_weights
        )

        self.biases = (
            self.biases
            -
            learning_rate
            *
            d_L_d_biases
        )

        return d_L_d_inputs.reshape(
            self.last_input_shape
        )