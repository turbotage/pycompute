

import time
import sys
import os

import numpy as np
import cupy as cp

import pycompute.cuda.sigpy.fourier_linops as fulinops
import pycompute.cuda.sigpy.fourier as fourier
import pycompute.cuda.sigpy.linop as linop


class Fourier_WReSL0(Alg):
	def __init__(self, y, mps, coord):

		# Sense operator

		self.ishape = mps.shape[1:]

		
		S = linop.Multiply(self.ishape, mps)

		S = fulinops.NUFFT(S.oshape, coord)

		self.sense_op = S

	def full_gradient():
		pass

	def update():
		pass




