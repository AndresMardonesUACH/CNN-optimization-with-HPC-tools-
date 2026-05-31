import time

from torchvision import datasets

import jax
import jax.numpy as jnp

from model import CNN


def load_cifar10():

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )

    x_train = jnp.asarray(
        train_dataset.data,
        dtype=jnp.float32
    ) / 255.0

    y_train = jnp.asarray(
        train_dataset.targets,
        dtype=jnp.int32
    )

    return (
        x_train,
        y_train
    )


x_train, y_train = load_cifar10()

model = CNN()


output = model.conv.forward(
    x_train[0]
)

jax.block_until_ready(output)

start = time.time()

for i in range(500):
    output = model.conv.forward(
        x_train[i]
    )

jax.block_until_ready(output)

end = time.time()

total = end - start

print(
    "Device:",
    jax.devices()[0]
)

print(
    f"Total time: {total:.4f} s"
)

print(
    f"Average per image: "
    f"{total/500:.6f} s"
)