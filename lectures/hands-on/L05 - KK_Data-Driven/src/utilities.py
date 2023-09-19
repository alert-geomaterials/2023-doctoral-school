# -*- coding: utf-8 -*-
"""
Various useful utilities

@author: Konstantinos Karapiperis
"""
import numpy as np

dim = 2

def convert_to_voigt_idx(node_idx, dof_idx):
	"""
	Returns voigt index corresponding to standard index
	"""
	return node_idx * dim + dof_idx

def convert_to_standard_idx(voigt_idx):
	"""
	Returns standard index corresponding to voigt index
	"""
	return divmod(voigt_idx, dim)

def convert_to_voigt_tensor(tensor):
	"""
	Returns same tensor reshaped in Voigt form
	"""
	voigt_tensor = np.zeros(tensor.shape[0]*tensor.shape[1])
	for dof_idx_i in range(dim):
		for dof_idx_j in range(dim):
			voigt_idx = convert_to_voigt_idx(dof_idx_i,dof_idx_j)
			voigt_tensor[voigt_idx] = tensor[dof_idx_i,dof_idx_j]
	return voigt_tensor

def convert_to_standard_tensor(voigt_tensor):
	"""
	Does the inverse operation as above
	"""
	tensor = np.zeros((int(np.sqrt(voigt_tensor.shape[0])),
					   int(np.sqrt(voigt_tensor.shape[0]))))
	for voigt_idx in range(dim*dim):
		std_idx = convert_to_standard_idx(voigt_idx)
		tensor[std_idx[0],std_idx[1]] = voigt_tensor[voigt_idx]
	return tensor

def convert_voigt_to_reduced_voigt(voigt_tensor):
	"""
	Returns reduced voigt tensor
	"""
	red_voigt_idx = [0,3,1]
	return voigt_tensor[:,red_voigt_idx]

def get_trace_of_reduced_voigt_tensor(vector):
	"""
	Returns trace of reduced Voigt tensor
	"""
	return np.sum(vector[:,:dim],axis=1)

def get_trace_of_squared_reduced_voigt_tensor(vector):
	"""
	Returns trace of reduced Voigt tensor
	"""
	trace_sqrd = np.zeros(vector.shape[0])
	for i in range(dim):
		trace_sqrd += vector[:,i]*vector[:,i]
	for i in range(dim,3*(dim-1)):
		trace_sqrd += 2*vector[:,i]*vector[:,i]
	return trace_sqrd
