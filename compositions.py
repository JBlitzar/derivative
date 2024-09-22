from modules import *
from extra import *



Tan = Composite.fromFunctional(Divide(Sin(X),Cos(X)), "Tan")
Cot = Composite.fromFunctional(Divide(Cos(X),Sin(X)), "Cot")
Sec = Composite.fromFunctional(Divide(1,Cos(X)), "Sec")
Csc = Composite.fromFunctional(Divide(1,Sin(X)), "Csc")


Inverse = Composite.fromFunctional(Divide(1,X), "Inverse")
Square = Composite.fromFunctional(Multiply(X,X), "Inverse")

Log10 = Composite.fromFunctional(Divide(Ln(X),Constant(Ln(Constant(10))(Constant(0)))), "Log10")

if __name__ == "__main__":
    expr = Tan(X())

    print(expr)

    print(expr.derivative()(1))

    print(expr.derivative().simplify()) # messy bc derived from quotient rule
