from layers.conv import ConvLayer
from layers.relu import ReLU
from layers.pool import MaxPool
from layers.dense import Dense
from layers.softmax import Softmax
import numpy as np


class CNN:

    def __init__(self):

        self.conv = ConvLayer(8)

        self.relu = ReLU()

        self.pool = MaxPool()

        self.dense = Dense(15 * 15 * 8, 10)

        self.softmax = Softmax()

    def forward(self, image):

        output = self.conv.forward(image)

        output = self.relu.forward(output)

        output = self.pool.forward(output)

        output = self.dense.forward(output)

        output = self.softmax.forward(output)

        return output

    def train(self, image, label, learning_rate=0.005):

        # ========= FORWARD =========

        output = self.forward(image)

        # ========= LOSS =========

        loss = -np.log(output[label] + 1e-12)

        # ========= ACCURACY =========

        accuracy = 1 if np.argmax(output) == label else 0

        # ========= INITIAL GRADIENT =========

        gradient = output.copy()

        gradient[label] -= 1

        # ========= BACKWARD =========

        gradient = self.dense.backward(
            gradient,
            learning_rate
        )

        gradient = self.pool.backward(gradient)

        gradient = self.relu.backward(gradient)

        self.conv.backward(
            gradient,
            learning_rate
        )

        return loss, accuracy