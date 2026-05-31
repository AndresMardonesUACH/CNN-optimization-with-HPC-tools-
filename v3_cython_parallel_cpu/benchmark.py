import time

from torchvision import datasets
from model import CNN
import numpy as np


def load_cifar10():

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )

    x_train = np.array(
        train_dataset.data,
        dtype=np.float32
    ) / 255.0

    y_train = np.array(
        train_dataset.targets,
        dtype=np.int32
    )
    

    return x_train, y_train


x_train, y_train = load_cifar10()

model = CNN()

start = time.time()

for i in range(500):
    model.train(x_train[i], y_train[i])

end = time.time()

total = end - start

print(f"Total time: {total:.4f} s")

print(f"Average per image: {total/500:.6f} s")