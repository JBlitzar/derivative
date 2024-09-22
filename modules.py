from typing import Type, Callable
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
    


    
    
class Composite(Expression):
    def __init__(self,f,g) -> None:
        super().__init__()

        self.f = f
        self.g = g

    def __call__(self, x: float) -> float:
        return self.f(self.g(x))
    
    
    def derivative(self) -> Type[Expression]:
        return Multiply(Composite(self.f.derivative(),self.g),self.g.derivative())
    
    def __repr__(self):
        return f"[({self.f}) of ({self.g})]"
    
    def simplify(self):
        return self
    
    @classmethod
    def fromFunctional(cls, func: Expression, name: str = "CompositeExpression") -> Type["Expression"]:
        class CompositeExpression(Expression):
            def __init__(self, inner_expr: Expression) -> None:
                super().__init__()
                self.inner_expr = inner_expr

            def __call__(self, x: float) -> float:

                return func(self.inner_expr)(x)
            
            def derivative(self) -> "Expression":
                return Multiply(Composite(func.derivative(), (self.inner_expr).simplify()), self.inner_expr.derivative().simplify())
            
            def __repr__(self) -> str:
                return f"{name}({self.inner_expr})"

        return CompositeExpression
    



class X(Expression):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: float) -> float:
        return x
    
    @staticmethod
    def derivative() -> Type[Expression]:
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
        return Divide(Subtract(Multiply(self.a.derivative(), self.b), Multiply(self.a, self.b.derivative())), Multiply(self.b, self.b))
    
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



if __name__ == "__main__":
    expr = Divide(Constant(1), X())

    print(expr.derivative()(1))
    print(expr.derivative().simplify())