from modules import *
import math



class PolynomialExponent(Expression):
    def __init__(self, exponent: float, baseExpression: Expression = X) -> None:
        super().__init__()


        self.exponent = exponent
        self.baseExpression = baseExpression

    def __call__(self, x: float) -> float:
        return  self.baseExpression(x) ** self.exponent
    
    
    def derivative(self) -> Type[Expression]:
        return Multiply(self.exponent, Multiply(self.baseExpression.derivative(), PolynomialExponent(self.exponent - 1, self.baseExpression)))
    
    def __repr__(self):
        return f"({self.baseExpression.__repr__()} ^ {self.exponent.__repr__()})"
    

    def simplify(self):
        self.baseExpression = self.baseExpression.simplify()
        if isinstance(self.baseExpression, Constant):
            return Constant(self.baseExpression ** self.exponent).simplify()

        return PolynomialExponent(self.exponent, self.baseExpression.simplify())
    
class EToTheF(Expression):
    def __init__(self, f: Expression = X) -> None:
        super().__init__()

        self.f = f

    def __call__(self, x: float) -> float:
        return math.e ** self.f(x)
    
    
    def derivative(self) -> Type[Expression]:
        return Multiply(self.f, self)
    
    def __repr__(self):
        return "e^X"
    
    def simplify(self):
        if isinstance(self.f, Constant) :
            return Constant(math.e ** self.exponent).simplify()

        return EToTheF(self.f.simplify())
    


class Sin(Expression):
    def __init__(self, f: Expression = X) -> None:
        super().__init__()

        self.f = f

    def __call__(self, x: float) -> float:
        return math.sin(x)
    
    
    def derivative(self) -> Type[Expression]:
        return Multiply(Cos(self.f), self.f.derivative())
    
    def __repr__(self):
        return f"sin({self.f})"
    
    def simplify(self):
        if isinstance(self.f, Constant) :
            return Constant(math.sin(self.f.k)).simplify()

        return Sin(self.f.simplify())
    
class Cos(Expression):
    def __init__(self, f: Expression = X) -> None:
        super().__init__()

        self.f = f

    def __call__(self, x: float) -> float:
        return math.cos(x)
    
    
    def derivative(self) -> Type[Expression]:
        return Multiply(Multiply(Constant(-1.0), Sin(self.f)), self.f.derivative())
    
    def __repr__(self):
        return f"cos({self.f})"
    
    def simplify(self):
        if isinstance(self.f, Constant) :
            return Constant(math.cos(self.f.k)).simplify()

        return Cos(self.f.simplify())