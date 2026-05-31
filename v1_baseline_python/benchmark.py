import time

from dataset import load_cifar10

from model import CNN


x_train, y_train, _, _ = load_cifar10()

model = CNN()

start = time.time()

for i in range(500):
    model.conv.forward(x_train[i])

end = time.time()

total_time = end - start

print(
    f"Total time: {total_time:.4f} s"
)

print(
    f"Average per image: "
    f"{total_time / 500:.6f} s"
)