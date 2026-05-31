from dataset import load_cifar10
from model import CNN
import time

x_train, y_train, _, _ = load_cifar10()


model = CNN()

for epoch in range(3):

    print(f"\nEpoch {epoch + 1}")

    loss = 0
    num_correct = 0

    for i in range(100):

        l, acc = model.train(
            x_train[i],
            y_train[i]
        )

        loss += l
        num_correct += acc

        if i % 10 == 0:

            print(
                f"Step {i} | "
                f"Avg Loss: {loss / (i + 1):.3f} | "
                f"Accuracy: {num_correct / (i + 1):.3f}"
            )