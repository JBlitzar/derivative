from modules import *
from extra import *
from compositions import *

class Parameter(Expression):
    def __init__(self, k: float) -> None:
        super().__init__()
        self.grad = 0.0
        self.k = k

    def __call__(self, x: float) -> float:
        return self.k
    
    def derivative(self) -> Expression:
        return Constant(1)

    def __repr__(self):
        return f"Parameter({self.k})"

    def update(self, lr: float) -> None:
        self.k -= lr * self.grad

class Linear(Expression):
    def __init__(self, weight: Parameter, bias: Parameter) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, x: float) -> float:
        return self.weight(x) * x + self.bias(x)  # w*x + b

    def derivative(self) -> Expression:
        return Add(Multiply(self.weight.derivative(), X()), self.bias.derivative())

    def __repr__(self):
        return f"(Linear(w={self.weight.__repr__()}, b={self.bias.__repr__()}))"
    
class MSELoss(Expression):
    def __init__(self, prediction: Expression, target: float) -> None:
        super().__init__()
        self.prediction = prediction
        self.target = Constant(target)

    def __call__(self, x: float) -> float:
        return (self.prediction(x) - self.target(x)) ** 2

    def derivative(self) -> Expression:
        return Multiply(Constant(2), Subtract(self.prediction, self.target))
    
    def __repr__(self):
        return f"MSELoss(pred={self.prediction}, target={self.target})"

