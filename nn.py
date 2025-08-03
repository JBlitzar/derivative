from modules import *
from extra import *
from compositions import *
import math


class Parameter:
    def __init__(self, value: float) -> None:
        self.value = value  # The value of the parameter (the actual number)
        self._grad = 0.0  # Gradient with respect to this parameter
        self._parents = []  # Parents and their local gradients
        self._op = None  # Store the operation ('add', 'mul', etc.) for tracking
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
        elif isinstance(other, (int, float)):
            out = Parameter(self.value + other)
            out._parents = [(self, 1)]  # Local gradient w.r.t self only
            out._op = "add_scalar"
            return out
        else:
            raise ValueError("Operand must be a Parameter or a number")

    def __sub__(self, other):
        if isinstance(other, Parameter):
            out = Parameter(self.value - other.value)
            out._parents = [(self, 1), (other, -1)]  # Local gradients w.r.t inputs
            out._op = "sub"
            return out
        elif isinstance(other, (int, float)):
            out = Parameter(self.value - other)
            out._parents = [(self, 1)]  # Local gradient w.r.t self only
            out._op = "sub_scalar"
            return out
        else:
            raise ValueError("Operand must be a Parameter or a number")

    def __mul__(self, other):
        if isinstance(other, Parameter):
            out = Parameter(self.value * other.value)
            out._parents = [(self, other.value), (other, self.value)]  # Local gradients
            out._op = "mul"
            return out
        elif isinstance(other, (int, float)):
            out = Parameter(self.value * other)
            out._parents = [(self, other)]  # Local gradient w.r.t self
            out._op = "mul_scalar"
            return out
        else:
            raise ValueError("Operand must be a Parameter or a number")

    def __rmul__(self, other):
        # Right-hand multiplication: other * self
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise ValueError("Operand must be a number")

    def __truediv__(self, other):
        if isinstance(other, Parameter):
            out = Parameter(self.value / other.value)
            out._parents = [
                (self, 1 / other.value),
                (other, -self.value / (other.value**2)),
            ]  # Chain rule for division
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

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            out = Parameter(self.value**other)
            out._parents = [
                (self, other * (self.value ** (other - 1)))
            ]  # d/dx(x^n) = n*x^(n-1)
            out._op = "pow"
            return out
        else:
            raise ValueError("Exponent must be a number")

    def exp(self):
        out = Parameter(math.exp(self.value))
        out._parents = [(self, math.exp(self.value))]  # d/dx(e^x) = e^x
        out._op = "exp"
        return out

    def log(self):
        out = Parameter(math.log(self.value))
        out._parents = [(self, 1 / self.value)]  # d/dx(ln(x)) = 1/x
        out._op = "log"
        return out

    def tanh(self):
        t = math.tanh(self.value)
        out = Parameter(t)
        out._parents = [(self, 1 - t**2)]  # d/dx(tanh(x)) = 1 - tanh^2(x)
        out._op = "tanh"
        return out

    def relu(self):
        out = Parameter(max(0, self.value))
        out._parents = [
            (self, 1 if self.value > 0 else 0)
        ]  # d/dx(relu(x)) = 1 if x > 0 else 0
        out._op = "relu"
        return out


class Linear:
    def __init__(self, weight: Parameter, bias: Parameter) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, x: float) -> float:
        return self.weight * x + self.bias  # w*x + b


class ReLU:
    def __call__(self, x):
        if isinstance(x, Parameter):
            return x.relu()  # Use the relu method from Parameter class
        else:
            return max(0, x)  # For float inputs

    def backward(self, grad_output: float) -> None:
        # Backward pass for ReLU
        if grad_output > 0:
            self._grad = grad_output
        else:
            self._grad = 0.0


class Sequential:
    def __init__(self, *layers) -> None:
        self.layers = layers

    def forward(self, x):
        """
        Forward pass through all layers.
        Handles both scalar and vector inputs.
        """
        current = x
        for layer in self.layers:
            if isinstance(layer, ReLU):
                # ReLU needs special handling for vectors
                if isinstance(current, list):
                    current = [layer(item) for item in current]
                else:
                    current = layer(current)
            else:
                current = layer(current)
        return current

    def backward(self, grad_output: float) -> None:
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                layer.backward(grad_output)
            grad_output = layer._grad  # Update gradient for next layer

    def parameters(self):
        """Get all parameters from all layers"""
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
            elif hasattr(layer, "weight") and hasattr(layer, "bias"):
                # Handle simple Linear layer
                params.extend([layer.weight, layer.bias])
        return params

    def zero_grad(self):
        """Reset gradients for all layers"""
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()
            elif hasattr(layer, "weight") and hasattr(layer, "bias"):
                # Handle simple Linear layer
                layer.weight.zero_grad()
                layer.bias.zero_grad()


class Dense:
    """
    A dense (fully connected) layer similar to PyTorch's Linear layer.
    Handles matrix operations for multiple inputs and outputs.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize weights with Xavier/Glorot initialization
        import random

        limit = (6.0 / (in_features + out_features)) ** 0.5

        # Create weight matrix as list of lists of Parameters
        self.weights = []
        for i in range(out_features):
            row = []
            for j in range(in_features):
                # Random initialization between -limit and +limit
                weight_val = random.uniform(-limit, limit)
                row.append(Parameter(weight_val))
            self.weights.append(row)

        # Initialize bias parameters
        if self.use_bias:
            self.biases = [Parameter(0.0) for _ in range(out_features)]
        else:
            self.biases = None

    def __call__(self, x):
        """
        Forward pass: y = Wx + b
        x can be either a list of floats/Parameters or a single float/Parameter
        """
        # Handle single input (convert to list)
        if isinstance(x, (float, int, Parameter)):
            x = [x]

        # Handle case where input is shorter than expected
        if len(x) < self.in_features:
            # Pad with zeros
            x = list(x) + [0.0] * (self.in_features - len(x))
        elif len(x) > self.in_features:
            # Truncate
            x = x[: self.in_features]

        outputs = []

        # Compute each output: out_i = sum(w_ij * x_j) + b_i
        for i in range(self.out_features):
            output = Parameter(0.0)  # Start with zero

            # Matrix multiplication: sum over input features
            for j in range(self.in_features):
                if isinstance(x[j], Parameter):
                    output = output + self.weights[i][j] * x[j]
                else:
                    output = output + self.weights[i][j] * x[j]

            # Add bias if enabled
            if self.use_bias:
                output = output + self.biases[i]

            outputs.append(output)

        # Return single Parameter if output_size is 1, otherwise return list
        if self.out_features == 1:
            return outputs[0]
        else:
            return outputs

    def parameters(self):
        """Return all parameters for optimization"""
        params = []
        for row in self.weights:
            params.extend(row)
        if self.use_bias:
            params.extend(self.biases)
        return params

    def zero_grad(self):
        """Reset gradients for all parameters"""
        for param in self.parameters():
            param.zero_grad()

    def __repr__(self):
        return f"Dense(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"
