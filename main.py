import unittest
import math
from modules import *
from extra import *
from compositions import *


if __name__ == "__main__":
    expr = PolynomialExponent(Constant(2),X())

    print(expr.derivative().simplify())

    print(expr.derivative()(1))
    
    