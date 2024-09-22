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
        return f"e^{self.f}"
    
    def simplify(self):
        if isinstance(self.f, Constant) :
            return Constant(math.e ** self.exponent).simplify()

        return EToTheF(self.f.simplify())
class Ln(Expression):
    def __init__(self, f: Expression = X) -> None:
        super().__init__()

        self.f = f

    def __call__(self, x: float) -> float:
        return math.log(self.f(x))
    
    
    def derivative(self) -> Type[Expression]:
        return Multiply(Divide(1,self.f), self.f.derivative())
    
    def __repr__(self):
        return f"ln({self.f})"
    
    def simplify(self):
        if isinstance(self.f, Constant) :
            return Constant(math.log(self.f.k)).simplify()

        return Ln(self.f.simplify())


class FToTheG(Expression):
    def __init__(self, f: Expression = X, g: Expression = X) -> None:
        super().__init__()

        self.f = f
        self.g = g

    def __call__(self, x: float) -> float:
        return self.f(x) ** self.g(x)
    
    #https://www.wolframalpha.com/input?i=derivative+f%28x%29%5Eg%28x%29
    def derivative(self) -> Type[Expression]:
        f_deriv = self.f.derivative()
        g_deriv = self.g.derivative()
        
        # Construct the derivative using the formula
        first_term = Multiply(self.g, f_deriv)  # g(x) f'(x)
        second_term = Multiply(Multiply(self.f, Ln(self.f)), g_deriv)  # f(x) log(f(x)) g'(x)
        inner_derivative = Add(first_term, second_term)  # g(x) f'(x) + f(x) log(f(x)) g'(x)
        
        # Outer derivative part is f(x)^(g(x) - 1)
        outer_derivative = PolynomialExponent(self.g - 1, self.f)
        
        # Full derivative: f(x)^(g(x) - 1) * (g(x) f'(x) + f(x) log(f(x)) g'(x))
        return Multiply(outer_derivative, inner_derivative)
    
    def __repr__(self):
        return f"{self.f} of ({self.g} of (x))"
    
    def simplify(self):
        return self
   

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