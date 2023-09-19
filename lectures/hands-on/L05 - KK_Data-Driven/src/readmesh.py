# -*- coding: utf-8 -*-
"""
Classes for reading mesh data and boundary conditions

@author: Konstantinos Karapiperis
"""

import numpy as np

class EssentialBCs():

    def __init__(self, ebc_dofs, ebc_vals):

        self.dofs = ebc_dofs
        self.vals = np.zeros_like(ebc_vals)
        self.final_vals = np.array(ebc_vals)

    def scale(self, scaling_factor):
        '''
        Scales all imposed values 
        '''
        self.vals = scaling_factor * self.final_vals

class Mesh():

    '''
    Output:
    - get_coords:             get coordinates of nodes 
    - get_conn:               get connectivity of elements
    - get_EBCs:               get EBC nodes and applied disp
    '''
    
    def __init__(self, coord_filename, conn_filename, ebc_filename):
        
        self.coord_filename = coord_filename
        self.conn_filename = conn_filename
        self.ebc_filename = ebc_filename
        self.n_dof = 2

    def get_coords(self):
        '''
        Node coordinates 
        '''
        coord = np.loadtxt(self.coord_filename)
        return len(coord[:,0]), coord

    def get_conn(self):
        '''
        Connectivity data for quadrilateral elements
        '''
        conn = np.loadtxt(self.conn_filename, dtype=int)
        return len(conn), conn
        
    def get_ebcs(self):
        '''
        Essential Boundary conditions
        node id, dof id, value
        '''
        ebc_data = np.loadtxt(self.ebc_filename)
        ebc_dofs = []
        ebc_vals = []
        
        for i in range(len(ebc_data)):
            ebc_dofs.append(int(ebc_data[i,0]*self.n_dof+ebc_data[i,1]))
            ebc_vals.append(ebc_data[i,2])

        return EssentialBCs(ebc_dofs,ebc_vals)