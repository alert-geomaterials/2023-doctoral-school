# -*- coding: utf-8 -*-
"""
Quadrilateral element class

@author: Konstantinos Karapiperis
"""
import sys
import numpy as np

class QuadElement():
    
    def __init__(self, dim, num):

        self.dim = dim
        self.num = num

        if self.dim != 2:
            sys.exit('Wrong number of dimensions!')
        if self.num != 4:
            sys.exit('Wrong number of gauss points!')

    def compute_N(self, point):
        """
        Evaluates the shape functions at given point
        """
        shape_functions = np.zeros(self.num)
        shape_functions[0] = 1./4*(1-point[0])*(1-point[1])
        shape_functions[1] = 1./4*(1+point[0])*(1-point[1]);
        shape_functions[2] = 1./4*(1+point[0])*(1+point[1]);
        shape_functions[3] = 1./4*(1-point[0])*(1+point[1]);
        return shape_functions

    def compute_dN(self, point):
        """
        Evaluates the shape function derivatives at given point
        """
        shape_function_derivatives = np.zeros((self.num,self.dim))
        shape_function_derivatives[0][0] = -1./4*(1-point[1])
        shape_function_derivatives[0][1] = -1./4*(1-point[0])
        shape_function_derivatives[1][0] = 1./4*(1-point[1])
        shape_function_derivatives[1][1] = -1./4*(1+point[0])
        shape_function_derivatives[2][0] = 1./4*(1+point[1])
        shape_function_derivatives[2][1] = 1./4*(1+point[0])
        shape_function_derivatives[3][0] = -1./4*(1+point[1])
        shape_function_derivatives[3][1] = 1./4*(1-point[0])

        return shape_function_derivatives