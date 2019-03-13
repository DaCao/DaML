import unittest
import numpy as np
import pickle
import pkg_resources

from LinearAlgebra import tools


class LinearAlgebraToolsTest(unittest.TestCase):
    def setUp(self):
        pass


    def test_functions(self):

        tools.LUdecomposition()


        tools.determinant()


        tools.inv_gaus()