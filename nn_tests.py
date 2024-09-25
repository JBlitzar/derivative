from nn import *


if __name__ == "__main__":
    # Define a simple linear model: y = w*x + b
    w = Parameter(2.0)  # Initial weight
    b = Parameter(1.0)  # Initial bias
    model = Linear(w, b)

    # Target value and input
    target = 10.0
    x = 3.0

    # Define loss function
    loss = MSELoss(model, target)

    # Forward pass: compute the loss
    output = loss(x)
    print(f"Loss: {output}")

    # Backward pass: compute the gradients
    d_loss = loss.derivative()
    print(f"Loss derivative: {d_loss}")

    # Compute gradients for the parameters
    # We need to evaluate the loss derivative with respect to the input (x)
    w.grad = d_loss(x) * x  # Gradient w.r.t weight
    b.grad = d_loss(x)      # Gradient w.r.t bias

    print(f"w.grad: {w.grad}, b.grad: {b.grad}")

    # Update parameters (gradient descent)
    lr = 0.01
    w.update(lr)
    b.update(lr)

    print(f"Updated weight: {w}, Updated bias: {b}")
