import unittest
from osea import *

class TestMathOperations(unittest.TestCase):
    def test_softmax(self):
        # use assertAlmostEqual for float comparing to avoid minor difference between 2 values
        result = softmax([(0, 0.12), (1, 1.22), (3, -0.23), (5, 0.87)])
        self.assertEqual(result[0][0], 1)
        self.assertAlmostEqual(result[0][1], 0.4401157667024683)
        self.assertEqual(result[1][0], 5)
        self.assertAlmostEqual(result[1][1], 0.3101443388926493)

    def test_get_filtered_predictions(self):
        result = get_filtered_predictions([0.2, 0.3, 1.12, 1.78], [2, 3])
        self.assertEqual(result, [(2, 1.12), (3, 1.78)])


if __name__ == '__main__':
    unittest.main()
