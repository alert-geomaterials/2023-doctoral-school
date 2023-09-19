# -*- coding: utf-8 -*-
"""
Material point class

@author: Konstantinos Karapiperis
"""
import numpy as np
from utilities import *

class MaterialPoint():
    
    def __init__(self, mp_id, density, thickness, nodal_positions, connectivity, quad_element, quad_rule, qp_index, mat_data_set, C, lamda, mu):
        
        self.mp_id = mp_id
        self.density = density
        self.thickness = thickness
        self.nodal_positions = nodal_positions
        self.connectivity = connectivity
        self.quad_element = quad_element
        self.quad_rule = quad_rule
        self.qp_index = qp_index
        self.mat_data_set = mat_data_set
        self.C = C
        self.lamda = lamda
        self.mu = mu
        self.distance = 1e8
        self.nodal_support = 4
        self.n_dof = 2
        self.mech_strain = np.zeros((self.n_dof,self.n_dof))
        self.mech_stress = np.zeros((self.n_dof,self.n_dof))
        self.mat_strain = np.zeros((self.n_dof,self.n_dof))
        self.mat_stress = np.zeros((self.n_dof,self.n_dof))

        # One-time calculations
        self.compute_shape_functions()
        self.k_e = self.compute_stiffness_matrix()
    
    def compute_shape_functions(self):
        """
        Compute jacobian of the mapping, shape functions and their derivatives
        """
        # Find location of quadrature point in parent domain
        quad_point = self.quad_rule.points[self.qp_index]

        # Compute shape function
        self.N = self.quad_element.compute_N(quad_point)

        # Compute shape function derivatives in parent domain
        dN_parent = self.quad_element.compute_dN(quad_point)

        # Initialize shape function derivatives in physical domain
        self.dN = np.zeros((self.nodal_support,self.n_dof))

        # Compute jacobian
        jacobian = np.zeros((self.n_dof,self.n_dof))
        for node_idx in range(self.nodal_support):
            pos = self.nodal_positions[self.connectivity[node_idx]]
            for dof_idx1 in range(self.n_dof):
                for dof_idx2 in range(self.n_dof):
                    jacobian[dof_idx1,dof_idx2] += pos[dof_idx2]*dN_parent[node_idx][dof_idx1]
        
        # Compute shape function derivatives
        for node_idx in range(self.nodal_support):
            self.dN[node_idx] = np.linalg.inv(jacobian).dot(dN_parent[node_idx])

        # Compute volume and nodal weights
        self.volume = np.abs(np.linalg.det(jacobian)) * self.quad_rule.weights[self.qp_index] * self.thickness
        self.nodal_weights = np.full(self.nodal_support, self.volume/self.nodal_support)

    def initialize_local_state(self, data_idx):
        '''
        Initialize material data idx
        '''
        self.data_idx = data_idx
        self.mat_strain = convert_to_standard_tensor(
            self.mat_data_set[data_idx,:self.n_dof*self.n_dof])
        self.mat_stress = convert_to_standard_tensor(
            self.mat_data_set[data_idx,self.n_dof*self.n_dof:])

    def compute_current_distance(self):
        '''
        Return weighted distance
        '''
        return self.distance * self.volume

    def compute_stress_based_forces(self):
        '''
        Computes forces due to material stress
        '''
        forces = []
        for node_idx in range(self.nodal_support):
            forces.append(-self.mat_stress.dot(self.dN[node_idx]) * self.volume)
        return forces

    def compute_strain_based_forces(self):
        '''
        Computes forces due to material strains
        '''
        forces = []
        strain_voigt = convert_to_voigt_tensor(self.mat_strain)
        stress_voigt = self.C.dot(strain_voigt)
        stress = convert_to_standard_tensor(stress_voigt)
        for node_idx in range(self.nodal_support):
            forces.append(stress.dot(self.dN[node_idx]) * self.volume)
        return forces

    def compute_stiffness_matrix(self):
        '''
        Computes local stiffness matrix
        '''
        k_e = np.zeros((self.n_dof*self.nodal_support, 
                        self.n_dof*self.nodal_support))        

        for node_idx_A in range(self.nodal_support):
            for node_idx_B in range(self.nodal_support):
                for i in range(self.n_dof):
                    voigt_idx_Ai = convert_to_voigt_idx(node_idx_A, i)
                    for k in range(self.n_dof):
                        voigt_idx_Bk = convert_to_voigt_idx(node_idx_B, k)
                        for j in range(self.n_dof):
                            voigt_idx_ij = convert_to_voigt_idx(i,j)
                            for l in range(self.n_dof):
                                voigt_idx_kl = convert_to_voigt_idx(k,l)
                                k_e[voigt_idx_Ai,voigt_idx_Bk] += \
                                    self.C[voigt_idx_ij,voigt_idx_kl] * \
                                    self.dN[node_idx_A][j] * self.dN[node_idx_B][l]

        return k_e * self.volume

    def compute_distance(self, dstrain_voigt, dstress_voigt):
        """
        Returns distance in phase space
        Note: The arguments should be reduced voigt tensors
        """
        dist = 0.5*self.lamda*np.power(get_trace_of_reduced_voigt_tensor(dstrain_voigt),2) \
             + self.mu*get_trace_of_squared_reduced_voigt_tensor(dstrain_voigt) \
             + 1./(4*self.mu)*get_trace_of_squared_reduced_voigt_tensor(dstress_voigt) \
             - self.lamda/(4*self.mu*(3*self.lamda+2*self.mu)) \
             * np.power(get_trace_of_reduced_voigt_tensor(dstress_voigt),2) 
        return dist

    def assign_local_states(self, displacements, lagrange_multipliers):
        """
        Updates mechanical stress and strain and then performs a search in the 
        dataset to update the material stress and strain 
        """
        # Update mechanical strain
        disp_gradient = np.zeros((self.n_dof, self.n_dof))
        for node_idx in range(self.nodal_support):
            disp_gradient += np.outer(displacements[node_idx],self.dN[node_idx])

        self.mech_strain = 0.5*(disp_gradient + disp_gradient.T)

        # Update mechanical stress
        lagrange_sum = np.zeros((self.n_dof, self.n_dof))
        for node_idx in range(self.nodal_support):
            lagrange_sum += np.outer(lagrange_multipliers[node_idx],self.dN[node_idx])

        stress_correction = self.C.dot(convert_to_voigt_tensor(lagrange_sum))
        self.mech_stress = self.mat_stress + convert_to_standard_tensor(stress_correction)

        # Update material stress and strain
        mech_strain_voigt = convert_to_voigt_tensor(self.mech_strain)
        mech_stress_voigt = convert_to_voigt_tensor(self.mech_stress)
        
        # Brute force seach in the dataset
        dists = self.compute_distance(
                convert_voigt_to_reduced_voigt(
                    mech_strain_voigt-self.mat_data_set[:,:self.n_dof*self.n_dof]),
                convert_voigt_to_reduced_voigt(
                    mech_stress_voigt-self.mat_data_set[:,self.n_dof*self.n_dof:]))
        idx_min = np.argmin(dists)

        # Check if the state changed 
        data_change = np.linalg.norm(
                        self.mat_data_set[self.data_idx,:self.n_dof*self.n_dof] 
                      - self.mat_data_set[idx_min,:self.n_dof*self.n_dof]) \
                    + np.linalg.norm(
                        self.mat_data_set[self.data_idx,self.n_dof*self.n_dof:] 
                      - self.mat_data_set[idx_min,self.n_dof*self.n_dof:]) 
        self.convergence_status = data_change < 1e-8
        self.data_idx = idx_min
        self.mat_strain = convert_to_standard_tensor(
            self.mat_data_set[idx_min,:self.n_dof*self.n_dof])
        self.mat_stress = convert_to_standard_tensor(
            self.mat_data_set[idx_min,self.n_dof*self.n_dof:])

        # Store the new distance
        dstrain_voigt = convert_to_voigt_tensor(self.mech_strain - self.mat_strain)
        dstress_voigt = convert_to_voigt_tensor(self.mech_stress - self.mat_stress)
        self.distance = self.compute_distance(
            convert_voigt_to_reduced_voigt(np.expand_dims(dstrain_voigt,axis=0)),
            convert_voigt_to_reduced_voigt(np.expand_dims(dstress_voigt,axis=0)))[0]
