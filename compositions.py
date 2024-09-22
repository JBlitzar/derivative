from modules import *
from extra import *



Tan = Composite.fromFunctional(Divide(Sin(X),Cos(X)), "Tan")
Cot = Composite.fromFunctional(Divide(Cos(X),Sin(X)), "Cot")
Sec = Composite.fromFunctional(Divide(1,Cos(X)), "Sec")
Csc = Composite.fromFunctional(Divide(1,Sin(X)), "Csc")


if __name__ == "__main__":
    expr = Tan(X())

    print(expr)

    print(expr.derivative()(1))

    print(expr.derivative().simplify()) # messy bc derived from quotient rule
