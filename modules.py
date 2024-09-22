from typing import Type
import math

class Expression:
    def __init__(self) -> None:
        pass

    def __call__(self,x: float) -> float:
        return 0.0
    
    def derivative(self):
        return Expression()
    
    def simplify(self):
        return self
    
    def __repr__(self):
        return "Expression"
    
    
class X(Expression):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: float) -> float:
        return x
    
    
    def derivative(self) -> Type[Expression]:
        return Constant(1)
    
    def __repr__(self):
        return "X"
    
    def simplify(self):
        return self

class Constant(Expression):
    def __init__(self, k: float) -> None:
        super().__init__()

        self.k = k

    def __call__(self, x: float) -> float:
        return self.k
    
    
    def derivative(self) -> Type[Expression]:
        return Constant(0.0)
    
    def __repr__(self):
        return f"{self.k.__repr__()}"
    
    def simplify(self):
        if isinstance(self.k, Constant):
            return Constant(self.k.k)
        return self
    
    def __add__(self, other):
        return Constant(self.k + other.k)
    
    def __sub__(self, other):
        return Constant(self.k - other.k)
    
    def __mul__(self, other):
        return Constant(self.k * other.k)
    
    def __div__(self, other):
        return Constant(self.k * other.k)
    
class Multiply(Expression):
    def __init__(self, a,b) -> None:
        super().__init__()


        self.a = a
        self.b = b

    def __call__(self, x: float) -> float:
        return self.a(x) * self.b(x)
    
    
    def derivative(self) -> Type[Expression]:
        return Add(Multiply(self.a.derivative(), self.b), Multiply(self.a, self.b.derivative()))
    
    def __repr__(self):
        return f"({self.a.__repr__()} * {self.b.__repr__()})"
    
    def simplify(self):
        if isinstance(self.a, Constant) and isinstance(self.b, Constant):
            return Constant(self.a * self.b).simplify()
        
        if isinstance(self.a, Constant) and self.a.k == 0.0 or isinstance(self.b, Constant) and self.b.k == 0.0:
            return Constant(0.0)

        return Multiply(self.a.simplify(), self.b.simplify())

class Divide(Expression):
    def __init__(self, a,b) -> None:
        super().__init__()


        self.a = a
        self.b = b

    def __call__(self, x: float) -> float:
        return self.a(x) / self.b(x)
    
    
    def derivative(self) -> Type[Expression]:
        return Divide(Subtract(Multiply(self.a.derivative(), self.b), Multiply(self.a, self.b.derivative())), PolynomialExponent(2, self.b))
    
    def __repr__(self):
        return f"({self.a.__repr__()} / {self.b.__repr__()})"
    
    def simplify(self):
        self.a = self.a.simplify()
        self.b = self.b.simplify()
        if isinstance(self.a, Constant) and isinstance(self.b, Constant):
            return Constant(self.a / self.b).simplify()

        return Divide(self.a.simplify(), self.b.simplify())

class Add(Expression):
    def __init__(self, a,b) -> None:
        super().__init__()


        self.a = a
        self.b = b

    def __call__(self, x: float) -> float:
        return self.a(x) + self.b(x)
    
    
    def derivative(self) -> Type[Expression]:
        return Add(self.a.derivative(), self.b.derivative())
    
    def __repr__(self):
        return f"({self.a.__repr__()} + {self.b.__repr__()})"
    
    def simplify(self):
        self.a = self.a.simplify()
        self.b = self.b.simplify()
        if isinstance(self.a, Constant) and isinstance(self.b, Constant):
            return Constant(self.a + self.b).simplify()

        return Add(self.a.simplify(), self.b.simplify())

class Subtract(Expression):
    def __init__(self, a,b) -> None:
        super().__init__()


        self.a = a
        self.b = b

    def __call__(self, x: float) -> float:
        return self.a(x) - self.b(x)
    
    
    def derivative(self) -> Type[Expression]:
        return Subtract(self.a.derivative(), self.b.derivative())
    
    def __repr__(self):
        return f"({self.a.__repr__()} - {self.b.__repr__()})"
    
    def simplify(self):
        self.a = self.a.simplify()
        self.b = self.b.simplify()
        if isinstance(self.a, Constant) and isinstance(self.b, Constant):
            
            return Constant(self.a - self.b).simplify()

        return Subtract(self.a, self.b)

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


if __name__ == "__main__":
    expr = Divide(Constant(1), X())

    print(expr.derivative()(1))
    print(expr.derivative().simplify())