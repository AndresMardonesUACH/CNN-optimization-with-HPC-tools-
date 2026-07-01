import ray
import numpy as np

class Conv3x3Ray:

    def __init__(self, num_filters=8):
        self.num_filters = num_filters

        self.filters = (
            np.random.randn(
                num_filters,
                3,
                3,
                3
            ).astype(np.float32)
            / 9
        )

        self.n_workers = min(4, num_filters)

        self.filter_indices = np.array_split(
            np.arange(self.num_filters),
            self.n_workers
        )  

    def forward(self, image):

        self.last_input = image

        self.last_input_ref = ray.put(image)

        filter_chunks = [
            self.filters[idx]
            for idx in self.filter_indices
        ]

        futures = [

            conv_filters_chunk.remote(
                self.last_input_ref,
                chunk
            )

            for chunk in filter_chunks
        ]

        outputs = ray.get(futures)

        return np.concatenate(
            outputs,
            axis=-1
        )
    
    def backward(self, d_L_d_out, learning_rate):

        futures = []

        for indices in self.filter_indices:
            d_L_d_out_chunk = d_L_d_out[:, :, indices]
            futures.append(
                backward_filters_chunk.remote(
                    self.last_input_ref,
                    d_L_d_out_chunk,
                )
            )

        results = ray.get(futures)

        d_L_d_filters = np.zeros_like(self.filters)

        for indices, grad_chunk in zip(self.filter_indices, results):
            d_L_d_filters[indices] = grad_chunk

        self.filters -= learning_rate * d_L_d_filters

        return None
    
@ray.remote
def conv_filters_chunk(image, filters_chunk):
    """
    image: (32,32,3)
    filters_chunk: (k,3,3,3)
    """

    h, w, _ = image.shape

    out_h = h - 2
    out_w = w - 2

    outputs = []

    for filt in filters_chunk:

        out = np.zeros((out_h, out_w), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):

                region = image[i:i+3, j:j+3]

                out[i, j] = np.sum(region * filt)

        outputs.append(out)

    return np.stack(outputs, axis=-1)

@ray.remote
def backward_filters_chunk(
    image,
    d_L_d_out_chunk,
):

    num_local_filters = d_L_d_out_chunk.shape[-1]

    grads = np.zeros(
        (num_local_filters, 3, 3, 3),
        dtype=np.float32
    )

    for local_f in range(num_local_filters):

        for i in range(d_L_d_out_chunk.shape[0]):
            for j in range(d_L_d_out_chunk.shape[1]):

                region = image[i:i+3, j:j+3]

                grads[local_f] += (
                    d_L_d_out_chunk[i, j, local_f]
                    * region
                )

    return grads