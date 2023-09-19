# -*- coding: utf-8 -*-
"""
Solver class

@author: Konstantinos Karapiperis
"""
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

class DistanceMinimizingSolver():
    
    def __init__(self, assembler, max_iterations):
        
        self.assembler = assembler
        self.max_iterations = max_iterations
        self.n_dof = assembler.n_dof

        # One-time calculations
        self.stiffness_matrix = assembler.assemble_stiffness_matrix()

    def compute_solution(self, essential_bcs, initial_guess, verbose=0):
        """
        Returns the converged solution starting from 
        given displacement initial guess
        """
        # Initialize displacements
        displacements = np.copy(initial_guess)

        # Impose essential bcs
        displacements[essential_bcs.dofs] = essential_bcs.vals

        # Update local states (strains and stresses)
        self.assembler.assign_local_states(displacements, np.zeros_like(displacements))

        # Apply bcs to the stiffness matrix
        stiffness_matrix = np.copy(self.stiffness_matrix)
        for dof in essential_bcs.dofs:
            stiffness_matrix[dof,:] = 0
            stiffness_matrix[dof,dof] = 1

        # Check for zero rows
        for row in stiffness_matrix:
            if np.linalg.norm(row) < 1e-4:
                sys.exit('Solver found zero row in the stiffness matrix. Exiting.')

        # Start iterating
        distances = []
        convergence = False
        iter = 0

        # Setup progress figure
        if verbose:
            fig = plt.figure()

        while convergence == False and iter < self.max_iterations:

            # Assemble forces
            stress_based_forces, strain_based_forces = \
                self.assembler.assemble_force_vectors()

            # Apply bcs to the force vectors
            for dof,val in zip(essential_bcs.dofs,essential_bcs.vals):
                strain_based_forces[dof] = val
                stress_based_forces[dof] = 0

            # Solve for the displacements and lagrange multipliers
            displacements = np.linalg.solve(stiffness_matrix, strain_based_forces)
            lagr_multipliers = np.linalg.solve(stiffness_matrix, stress_based_forces)

            # Assign new local states
            self.assembler.assign_local_states(displacements, lagr_multipliers)

            # Check for convergence and update iteration counter
            convergence = self.assembler.get_convergence_status()
            distances.append(self.assembler.compute_global_distance())
            iter += 1

            if verbose:
                plt.clf()
                plt.xlabel('Iterations')
                plt.ylabel('Global distance')
                plt.plot(range(iter), distances, '-o', c='r')
                display.clear_output(wait=True)
                display.display(plt.gcf())
                time.sleep(0.5)

        if verbose: plt.close()

        if iter == self.max_iterations:
            sys.exit('Reached maximum number of iterations. Exiting.')

        return displacements
