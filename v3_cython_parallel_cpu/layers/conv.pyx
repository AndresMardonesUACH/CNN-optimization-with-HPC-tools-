# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

from cython.parallel import prange, threadid

ctypedef cnp.float32_t DTYPE_t

cdef extern from "omp.h":
    int omp_get_num_threads()
    int omp_get_max_threads()

cdef inline DTYPE_t convolve_region(
    DTYPE_t[:, :, :] input_view,
    DTYPE_t[:, :, :, :] filters_view,
    int f,
    int i,
    int j
) nogil:

    cdef:
        int i2, j2, c
        DTYPE_t total = 0.0

    for i2 in range(3):
        for j2 in range(3):
            for c in range(3):

                total += (
                    input_view[
                        i + i2,
                        j + j2,
                        c
                    ]
                    *
                    filters_view[
                        f,
                        i2,
                        j2,
                        c
                    ]
                )

    return total


cdef class ConvLayer:

    cdef:
        int num_filters
        cnp.ndarray filters_np
        cnp.ndarray last_input_np

    def __init__(self, int num_filters):

        self.num_filters = num_filters

        self.filters_np = (
            np.random.randn(
                num_filters,
                3,
                3,
                3
            ).astype(np.float32) / 9
        )


    def forward(
        self,
        cnp.ndarray[DTYPE_t, ndim=3] input
    ):

        self.last_input_np = input

        cdef int h = input.shape[0]
        cdef int w = input.shape[1]

        cdef cnp.ndarray[DTYPE_t, ndim=3] output = np.zeros(
            (h - 2, w - 2, self.num_filters),
            dtype=np.float32
        )

        cdef:
            DTYPE_t[:, :, :] input_view = input
            DTYPE_t[:, :, :] output_view = output
            DTYPE_t[:, :, :, :] filters_view = self.filters_np

        cdef:
            int i, j, f

        # paralelizar sobre filtros
        for f in prange(
            self.num_filters,
            nogil=True,
            schedule='static'
        ):

            for i in range(h - 2):

                for j in range(w - 2):

                    output_view[i, j, f] = convolve_region(
                        input_view,
                        filters_view,
                        f,
                        i,
                        j
                    )

        return output


    def backward(
        self,
        cnp.ndarray d_L_d_out,
        float learning_rate
    ):


        d_L_d_out = np.asarray(
            d_L_d_out,
            dtype=np.float32
        )

        self.last_input_np = np.asarray(
            self.last_input_np,
            dtype=np.float32
        )

        self.filters_np = np.asarray(
            self.filters_np,
            dtype=np.float32
        )

        # -------------------------
        # grad filtros
        # -------------------------

        cdef cnp.ndarray[
            DTYPE_t,
            ndim=4
        ] d_L_d_filters = np.zeros(
            (
                self.num_filters,
                3,
                3,
                3
            ),
            dtype=np.float32
        )

        # -------------------------
        # tamaños
        # -------------------------

        cdef:
            int h = self.last_input_np.shape[0]
            int w = self.last_input_np.shape[1]

        # -------------------------
        # memoryviews
        # -------------------------

        cdef:
            DTYPE_t[:, :, :] grad_view = d_L_d_out

            DTYPE_t[:, :, :] input_view = self.last_input_np

            DTYPE_t[:, :, :, :] filters_grad_view = d_L_d_filters

        # -------------------------
        # índices
        # -------------------------

        cdef:
            int i
            int j
            int f
            int i2
            int j2
            int c

        # -------------------------
        # OpenMP
        # cada thread trabaja
        # un filtro distinto
        # -------------------------

        for f in prange(
            self.num_filters,
            nogil=True,
            schedule='static'
        ):

            for i in range(
                h - 2
            ):

                for j in range(
                    w - 2
                ):

                    for i2 in range(3):

                        for j2 in range(3):

                            for c in range(3):

                                filters_grad_view[
                                    f,
                                    i2,
                                    j2,
                                    c
                                ] += (

                                    grad_view[
                                        i,
                                        j,
                                        f
                                    ]

                                    *

                                    input_view[
                                        i + i2,
                                        j + j2,
                                        c
                                    ]
                                )

        # -------------------------
        # update pesos
        # fuera de nogil
        # -------------------------

        self.filters_np -= (
            learning_rate
            *
            d_L_d_filters
        )

        return None