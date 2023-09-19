# -*- coding: utf-8 -*-
"""
Quadrature rule class

@author: Konstantinos Karapiperis
"""
import sys
import numpy as np

class Quadrature():
    
    def __init__(self, dim, num):

        if dim != 2:
            sys.exit('Wrong number of dimensions!')
        if num != 4:
            sys.exit('Wrong number of gauss points!')

        self.points = np.zeros((num, dim))
        self.points[0] = [-0.577350269189626e0, -0.577350269189626e0]
        self.points[1] = [-0.577350269189626e0, 0.577350269189626e0]
        self.points[2] = [0.577350269189626e0, -0.577350269189626e0]
        self.points[3] = [0.577350269189626e0, 0.577350269189626e0]
        self.weights = np.array([1.,1.,1.,1.])
