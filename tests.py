import unittest
import math
from modules import *
from extra import *
from compositions import *

class TestExpressionDerivatives(unittest.TestCase):
    
    def test_constant_derivative(self):
        # d/dx(C) = 0 for any x
        expr = Constant(5)
        self.assertAlmostEqual(expr.derivative()(10), 0.0)

    def test_x_derivative(self):
        # d/dx(x) = 1 for any x
        expr = X()
        self.assertAlmostEqual(expr.derivative()(10), 1.0)

    def test_addition_derivative(self):
        # d/dx(x + 5) = 1 for any x
        expr = Add(X(), Constant(5))
        self.assertAlmostEqual(expr.derivative()(10), 1.0)

    def test_subtraction_derivative(self):
        # d/dx(x - 5) = 1 for any x
        expr = Subtract(X(), Constant(5))
        self.assertAlmostEqual(expr.derivative()(10), 1.0)

    def test_multiplication_derivative(self):
        # d/dx(x * x) = 2x
        expr = Multiply(X(), X())
        self.assertAlmostEqual(expr.derivative()(2), 4.0)
        self.assertAlmostEqual(expr.derivative()(3), 6.0)

    def test_division_derivative(self):
        # d/dx(1/x) = -1/x^2
        expr = Divide(Constant(1), X())
        self.assertAlmostEqual(expr.derivative()(2), -0.25)
        self.assertAlmostEqual(expr.derivative()(0.5), -4.0)

    def test_exponential_derivative(self):
        # d/dx(e^x) = e^x
        expr = EToTheF(X())
        self.assertAlmostEqual(expr.derivative()(0), math.exp(0))
        self.assertAlmostEqual(expr.derivative()(1), math.exp(1))

    def test_power_derivative(self):
        # d/dx(x^3) = 3x^2
        expr = FToTheG(Constant(3), X())
        self.assertAlmostEqual(expr.derivative()(2), 12.0)
        self.assertAlmostEqual(expr.derivative()(3), 27.0)

    def test_ln_derivative(self):
        # d/dx(ln(x)) = 1/x
        expr = Ln(X())

        self.assertAlmostEqual(expr.derivative()(2), 0.5)
        self.assertAlmostEqual(expr.derivative()(1), 1.0)

    def test_sin_derivative(self):
        # d/dx(sin(x)) = cos(x)
        expr = Sin(X())
        self.assertAlmostEqual(expr.derivative()(0), math.cos(0))
        self.assertAlmostEqual(expr.derivative()(math.pi / 2), math.cos(math.pi / 2))

    def test_cos_derivative(self):
        # d/dx(cos(x)) = -sin(x)
        expr = Cos(X())
        self.assertAlmostEqual(expr.derivative()(0), -math.sin(0))
        self.assertAlmostEqual(expr.derivative()(math.pi / 2), -math.sin(math.pi / 2))

    def test_tan_derivative(self):
        # d/dx(tan(x)) = sec^2(x)
        expr = Tan(X())
        self.assertAlmostEqual(expr.derivative()(0), 1.0)
        self.assertAlmostEqual(expr.derivative()(math.pi / 4), 2.0)

    def test_csc_derivative(self):
        # d/dx(csc(x)) = -csc(x)cot(x)
        expr = Csc(X())
        self.assertAlmostEqual(expr.derivative()(math.pi / 2), -1.0)
        self.assertAlmostEqual(expr.derivative()(math.pi / 4), -math.sqrt(2))

    def test_sec_derivative(self):
        # d/dx(sec(x)) = sec(x)tan(x)
        expr = Sec(X())
        self.assertAlmostEqual(expr.derivative()(0), 0.0)
        self.assertAlmostEqual(expr.derivative()(math.pi / 4), math.sqrt(2))

    def test_cot_derivative(self):
        # d/dx(cot(x)) = -csc^2(x)
        expr = Cot(X())
        self.assertAlmostEqual(expr.derivative()(math.pi / 4), -2.0)
        self.assertAlmostEqual(expr.derivative()(math.pi / 2), 0.0)

    def test_compound_expression_derivative(self):
        print("ITS TIME")
        # d/dx(3x^2 + 2x + 1) = 6x + 2
        expr = Add(Add(Multiply(Constant(3), Square(X)), Multiply(Constant(2), X())), Constant(1))
        d = expr.derivative()
        self.assertAlmostEqual(d(1), 8.0)
        self.assertAlmostEqual(expr.derivative()(2), 14.0)

if __name__ == "__main__":
    unittest.main()
