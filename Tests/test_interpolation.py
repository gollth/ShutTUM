import unittest
import numpy as np
from ShutTUM import Interpolation


class TestInterpolation(unittest.TestCase):

    def setUp(self):
        self.linear = Interpolation.linear
        self.slerp  = Interpolation.slerp
        self.cubic  = Interpolation.cubic
        self.lower  = np.array((0,2,10))
        self.upper  = np.array((1,4,110))
        self.middle = np.array((.5, 3, 60))

    def test_linear_for_lower_bound(self):
        self.assertEqual(self.linear(0, 1, 0), 0)
        self.assertEqual(self.linear(2, 180, 0), 2)
        self.assertEqual(self.linear(self.lower, self.upper, 0).tolist(), self.lower.tolist())

    def test_linear_for_upper_bound(self):
        self.assertEqual(self.linear(0, 1, 1), 1)
        self.assertEqual(self.linear(2, 180, 1), 180)
        self.assertListEqual(self.linear(self.lower, self.upper, 1).tolist(), self.upper.tolist())

    def test_linear_50_percent(self):
        self.assertEqual(self.linear(0, 1, 0.5), 0.5)
        self.assertEqual(self.linear(2, 182, 0.5), 92)
        self.assertListEqual(self.linear(self.lower, self.upper, .5).tolist(), self.middle.tolist())


    def test_cubic_interpolation_with_borders(self):
        # see http://www.paulinternet.nl/?page=bicubic
        a0, a, b, b0 = 2, 4, 2, 3

        def f(x): return 7./2.*x**3 - 11./2.*x**2 + 4

        for x in np.linspace(0,1, num=10):
            exp = f(x)
            act = self.cubic(a,b, x, a0, b0)
            self.assertAlmostEqual(exp, act)

    def test_cubic_interpolation_without_borders(self):
        a, b = 0, 1

        def f(x): return (a-b)*x**3 + 1.5*(b-a)*x**2 + (.5*b+.5*a)*x + a

        for x in np.linspace(0,1, num=16):
            exp = f(x)
            act = self.cubic(a,b,x)
            self.assertAlmostEqual(exp, act)


if __name__ == '__main__':
    unittest.main()
