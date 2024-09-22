from typing import Type
class Expression:
    def __init__(self) -> None:
        pass

    def __call__(self,x: float) -> float:
        return 0.0
    
    def derivative(self):
        return Expression()
    
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

class Constant(Expression):
    def __init__(self, k: float) -> None:
        super().__init__()


        self.k = k

    def __call__(self, x: float) -> float:
        return self.k
    
    
    def derivative(self) -> Type[Expression]:
        return Constant(0.0)
    
    def __repr__(self):
        return self.k.__repr__()
    
class MultipliedBy(Expression):
    def __init__(self, a,b) -> None:
        super().__init__()


        self.a = a
        self.b = b

    def __call__(self, x: float) -> float:
        return self.a(x) * self.b(x)
    
    
    def derivative(self) -> Type[Expression]:
        return Add(MultipliedBy(self.a.derivative(), self.b), MultipliedBy(self.a, self.b.derivative()))
    
    def __repr__(self):
        return f"({self.a.__repr__()} * {self.b.__repr__()})"

class DividedBy(Expression):
    def __init__(self, a,b) -> None:
        super().__init__()


        self.a = a
        self.b = b

    def __call__(self, x: float) -> float:
        return self.a(x) / self.b(x)
    
    
    def derivative(self) -> Type[Expression]:
        return DividedBy(Subtract(MultipliedBy(self.a.derivative(), self.b), MultipliedBy(self.a, self.b.derivative())), PolynomialExponent(2, self.b))
    
    def __repr__(self):
        return f"({self.a.__repr__()} / {self.b.__repr__()})"

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

class PolynomialExponent(Expression):
    def __init__(self, exponent: float, baseExpression: Expression = X) -> None:
        super().__init__()


        self.exponent = exponent
        self.baseExpression = baseExpression

    def __call__(self, x: float) -> float:
        return  self.baseExpression(x) ** self.exponent
    
    
    def derivative(self) -> Type[Expression]:
        return MultipliedBy(self.exponent, PolynomialExponent(self.exponent - 1))
    
    def __repr__(self):
        return f"({self.baseExpression.__repr__()} ^ {self.exponent.__repr__()})"
    



if __name__ == "__main__":
    expr = DividedBy(Constant(1), X())

    print(expr.derivative()(1))
    print(expr.derivative())