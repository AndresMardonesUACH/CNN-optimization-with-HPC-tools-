from torchvision import datasets
import numpy as np

def load_cifar10():

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True
    )

    x_train = np.array(train_dataset.data, dtype=np.float32) / 255.0
    y_train = np.array(train_dataset.targets)

    x_test = np.array(test_dataset.data, dtype=np.float32) / 255.0
    y_test = np.array(test_dataset.targets)

    return x_train, y_train, x_test, y_test