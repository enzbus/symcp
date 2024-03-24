"""Test symbolic array creation, combinations, operations, numpy values."""

from unittest import TestCase

import numpy as np

import symcp as cp


class TestSymArray(TestCase):
    """Test class for basic symbolic array operations."""

    def test_variable_create(self):
        """Simple test for variable creation and error checks."""

        x = cp.Variable()
        self.assertEqual(x.shape, ())

        x = cp.Variable(3)
        self.assertEqual(x.shape, (3,))

        # with self.assertRaises(ValueError):
        #     cp.Variable((3, 3))

        with self.assertRaises(ValueError):
            cp.Variable(3.3)

    def test_matmul(self):
        """Test matrix multiplication."""

        n, m = 5, 7
        mat = np.random.randn(n, m)
        x = cp.Variable(m)
        symarr = mat @ x
        self.assertEqual(symarr.shape, (n,))

        y = cp.Variable(n)
        symarr = y.T @ mat
        self.assertEqual(symarr.shape, (m,))

        symarr = y @ mat
        self.assertEqual(symarr.shape, (m,))

    def test_broadcasting(self):
        """Test numpy broadcasting."""

        x = cp.Variable()
        y = cp.Variable(3)

        symarr = x + y
        self.assertEqual(symarr.shape, (3,))

        c = cp.Parameter((3, 1))
        b = cp.Parameter((1, 4))
        symarr = c + b
        self.assertEqual(symarr.shape, (3, 4))
