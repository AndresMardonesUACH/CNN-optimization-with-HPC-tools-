import jax
import jax.numpy as jnp


class ConvLayer:

    def __init__(self, num_filters):

        self.num_filters = num_filters

        key = jax.random.PRNGKey(0)

        self.filters = (
            jax.random.normal(
                key,
                (
                    num_filters,
                    3,
                    3,
                    3
                )
            )
            / 9.0
        ).astype(jnp.float32)

    @staticmethod
    @jax.jit
    def _forward_jit(
        input_image,
        filters
    ):

        # (30,30,3,3,3)
        windows = jax.lax.conv_general_dilated_patches(
            lhs=input_image[None, ...],
            filter_shape=(3, 3),
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "OIHW", "NHWC")
        )

        # reshape patches:
        # (1,30,30,27) -> (30,30,3,3,3)
        windows = windows.reshape(
            30,
            30,
            3,
            3,
            3
        )

        output = jnp.tensordot(
            windows,
            filters,
            axes=([2, 3, 4], [1, 2, 3])
        )

        return output

    def forward(
        self,
        input_image
    ):

        input_image = jnp.asarray(
            input_image,
            dtype=jnp.float32
        )

        self.last_input = input_image

        return self._forward_jit(
            input_image,
            self.filters
        )

    def backward(
        self,
        d_L_d_out,
        learning_rate
    ):

        windows = jax.lax.conv_general_dilated_patches(
            lhs=self.last_input[None, ...],
            filter_shape=(3, 3),
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "OIHW", "NHWC")
        )

        windows = windows.reshape(
            30,
            30,
            3,
            3,
            3
        )

        d_L_d_filters = jnp.tensordot(
            d_L_d_out,
            windows,
            axes=([0, 1], [0, 1])
        )

        self.filters = (
            self.filters
            -
            learning_rate
            *
            d_L_d_filters
        )

        return None