

import time
import sys
import os

import numpy as np
import cupy as cp

import pycompute.cuda.sigpy.fourier_linops as fulinops
import pycompute.cuda.sigpy.fourier as fourier
import pycompute.cuda.sigpy.linop as linop


class Fourier_WReSL0():
	def __init__(self, y, x0, mps, coord, batch_size):

		# Sense operator

		self.ishape = mps.shape[1:]

		self.x = x0
		
		self.sense_op = linop.Multiply(self.ishape, mps)
		self.sense_op = fulinops.NUFFT(self.sense_op.oshape, coord) * self.sense_op

		self.sigma_max = 1
		self.sigma_min = 1e-3

		self.sigma_k = self.sigma_max

		self.batch_size = batch_size


	def sparsity_gradient(self):



	def update():
		pass




