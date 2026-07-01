import time
from torchvision import datasets
from model import CNN
import numpy as np
import argparse
from memory_profiler import memory_usage
import jax



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


x_train, y_train= load_cifar10()
model = CNN()

def benchmark_train(n_images):
    for i in range(n_images):
        model.train(x_train[i], y_train[i])

def benchmark_forward(n_images):
    for i in range(n_images):
        model.forward(x_train[i])

def benchmark_conv(n_images):
    for i in range(n_images):
        model.conv.forward(x_train[i])

BENCHMARKS = {
    "train": benchmark_train,
    "forward": benchmark_forward,
    "conv": benchmark_conv,
}


def benchmark_time(func, n_images):
    output = model.conv.forward(
        x_train[0]
    )
    jax.block_until_ready(output)
    start = time.perf_counter()

    func(n_images)

    end = time.perf_counter()
    jax.block_until_ready(output)
    total = end - start

    print("Device:", jax.devices()[0])
    print(f"Total time: {total:.4f} s")
    print(f"Average per image: {total / n_images:.6f} s")


def benchmark_memory(func, n_images):
    mem = memory_usage(
        (func, (n_images,)),
        interval=0.001,
    )
    print(
        f"Peak memory: "
        f"{max(mem) - mem[0]:.2f} MiB"
    )


def benchmark_energy(func, n_images):
    import pyRAPL
    pyRAPL.setup()
    measurement = pyRAPL.Measurement("benchmark")
    measurement.begin()
    func(n_images)
    measurement.end()

    print(
        f"Energy: "
        f"{measurement.result.pkg[0] / 1_000_000:.6f} J"
    )

MODES = {
    "tiempo": benchmark_time,
    "memoria": benchmark_memory,
    "energia": benchmark_energy,
}

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "modo",
        choices=["tiempo", "memoria", "energia"],
    )

    parser.add_argument(
        "funcion",
        choices=["train", "forward", "conv"],
    )

    parser.add_argument(
        "imagenes",
        type=int,
    )

    args = parser.parse_args()

    func = BENCHMARKS[args.funcion]

    MODES[args.modo](func, args.imagenes)


if __name__ == "__main__":
    main()