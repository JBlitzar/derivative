from nn import Sequential, Linear, ReLU, Parameter, Dense
import math
import random

if __name__ == "__main__":
    print("\n=== Testing Sequential Model with Dense Layers ===")

    network = Sequential(
        Dense(1, 5),
        ReLU(),
        Dense(5, 5),
        ReLU(),
        Dense(5, 2),
    )
    for i in range(1000):
        print(f"Iteration {i + 1}:")
        a = random.random()
        x_vec = [a]
        target_vec = [a**2, 3 * a + 3]

        output = network.forward(x_vec)

        loss_terms = [(output[i] - target_vec[i]) ** 2 for i in range(len(output))]
        total_loss = loss_terms[0]
        for i in range(1, len(loss_terms)):
            total_loss = total_loss + loss_terms[i]

        total_loss.backward()
        print(f"Total loss: {total_loss.value:.4f}")
        print(f"Output: [{', '.join([f'{o.value:.3f}' for o in output])}]")
        print(f"Target: [{', '.join([f'{t:.3f}' for t in target_vec])}]")

        learning_rate = 0.01

        for param in network.parameters():
            param.value -= learning_rate * param._grad

        network.zero_grad()
