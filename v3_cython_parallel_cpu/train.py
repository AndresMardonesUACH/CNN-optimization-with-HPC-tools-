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
        train_dataset.targets
    )

    return x_train, y_train


x_train, y_train = load_cifar10()

model = CNN()

for epoch in range(3):

    print(f"\nEpoch {epoch+1}")

    loss = 0
    correct = 0

    for i in range(100):

        l, acc = model.train(
            x_train[i],
            y_train[i]
        )

        loss += l
        correct += acc

        if i % 10 == 0:

            print(
                f"Step {i} | "
                f"Loss {loss/(i+1):.3f} | "
                f"Acc {correct/(i+1):.3f}"
            )