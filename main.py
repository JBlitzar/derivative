import unittest
import math
from modules import *
from extra import *
from compositions import *


if __name__ == "__main__":
    expr = Add(EToTheF(X()), Multiply(Constant(-2),X()))

    print(expr.derivative()(1))
    print(expr.derivative().simplify())