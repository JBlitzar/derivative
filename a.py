class Parameter:
    def __init__(self, value: float) -> None:
        self.value = value  # The value of the parameter (the actual number)
        self._grad = 0.0    # Gradient with respect to this parameter
        self._parents = []  # Parents and their local gradients
        self._op = None     # Store the operation ('add', 'mul', etc.) for tracking
        self._local_grads = []  # Store local gradients for chain rule

    def __repr__(self) -> str:
        return f"Parameter(value={self.value}, grad={self._grad})"
    
    # Forward pass for operations
    def __add__(self, other):
        if isinstance(other, Parameter):
            out = Parameter(self.value + other.value)
            out._parents = [(self, 1), (other, 1)]  # Local gradients w.r.t inputs
            out._op = "add"
            return out
        else:
            raise ValueError("Operand must be a Parameter")

    def __sub__(self, other):
        if isinstance(other, Parameter):
            out = Parameter(self.value - other.value)
            out._parents = [(self, 1), (other, -1)]  # Local gradients w.r.t inputs
            out._op = "sub"
            return out
        else:
            raise ValueError("Operand must be a Parameter")

    def __mul__(self, other):
        if isinstance(other, Parameter):
            out = Parameter(self.value * other.value)
            out._parents = [(self, other.value), (other, self.value)]  # Local gradients
            out._op = "mul"
            return out
        else:
            raise ValueError("Operand must be a Parameter")

    def __truediv__(self, other):
        if isinstance(other, Parameter):
            out = Parameter(self.value / other.value)
            out._parents = [(self, 1 / other.value), (other, -self.value / (other.value ** 2))]  # Chain rule for division
            out._op = "div"
            return out
        else:
            raise ValueError("Operand must be a Parameter")

    def __neg__(self):
        out = Parameter(-self.value)
        out._parents = [(self, -1)]  # Local gradient w.r.t input
        out._op = "neg"
        return out

    # Backward pass: accumulate gradients using chain rule
    def backward(self, grad: float = 1.0) -> None:
        """Recursively accumulate gradients by propagating backwards."""
        # Add the current gradient to this parameter
        self._grad += grad
        
        # Propagate to parents (if any)
        for parent, local_grad in self._parents:
            parent.backward(grad * local_grad)

    # Reset gradients (for the next iteration, typical in optimizers)
    def zero_grad(self) -> None:
        self._grad = 0.0

# Example usage
if __name__ == "__main__":
    # Example of a simple forward pass with two parameters
    w = Parameter(2.0)
    x = Parameter(3.0)
    b = Parameter(1.0)

    # Forward pass: y = w * x + b
    y = w * x + b

    print(f"Forward: y = {y.value}")  # Expected: 2 * 3 + 1 = 7

    # Backward pass: compute gradients w.r.t w, x, b
    y.backward()  # Starting from dL/dy = 1

    print(f"Gradient w.r.t w: {w._grad}")  # Expected: dL/dw = x = 3
    print(f"Gradient w.r.t x: {x._grad}")  # Expected: dL/dx = w = 2
    print(f"Gradient w.r.t b: {b._grad}")  # Expected: dL/db = 1

    # Reset gradients for the next iteration
    w.zero_grad()
    x.zero_grad()
    b.zero_grad()
