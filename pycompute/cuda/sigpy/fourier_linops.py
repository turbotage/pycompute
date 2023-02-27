
import cupy as cp
import numpy as np
import math

import pycompute.cuda.sigpy.linop as slinop
import pycompute.cuda.sigpy.fourier as sf

from pycompute.cuda.sigpy.linop import Linop, Identity, Resize, Multiply


class FFT(Linop):
	def __init__(self, shape, axes=None, center=True):
		self.axes = axes
		self.center = center

		super().__init__(shape, shape)


	def _apply(self, input):
		return sf.fft(input, axes=self.axes, center=self.center)

	def _adjoint_linop(self):
		return IFFT(self.ishape, axes=self.axes, center=self.center)

	def _normal_linop(self):
		return Identity(self.ishape)

class IFFT(Linop):
	def __init__(self, shape, axes=None, center=True):

		self.axes = axes
		self.center = center

		super().__init__(shape, shape)


	def _apply(self, input):
		return sf.ifft(input, axes=self.axes, center=self.center)

	def _adjoint_linop(self):
		return FFT(self.ishape, axes=self.axes, center=self.center)

	def _normal_linop(self):
		return Identity(self.ishape)

class NUFFT(Linop):
	def __init__(self, ishape, coord, oversamp=1.25, width=4, toeplitz=False):
		self.coord = coord
		self.oversamp = oversamp
		self.width = width
		self.toeplitz = toeplitz

		ndim = coord.shape[-1]

		oshape = list(ishape[:-ndim]) + list(coord.shape[:-1])

		super().__init__(oshape, ishape)


	def _apply(self, input):
		return sf.nufft(
			input, self.coord,
			oversamp=self.oversamp, width=self.width)

	def _adjoint_linop(self):
		return NUFFTAdjoint(self.ishape, self.coord,
							oversamp=self.oversamp, width=self.width)

	def _normal_linop(self):
		if self.toeplitz is False:
			return self.H * self

		ndim = self.coord.shape[-1]
		psf = sf.toeplitz_psf(self.coord, self.ishape, self.oversamp,
								   self.width)

		fft_axes = tuple(range(-1, -(ndim + 1), -1))

		R = Resize(psf.shape, self.ishape)
		F = FFT(psf.shape, axes=fft_axes)
		P = Multiply(psf.shape, psf)
		T = R.H * F.H * P * F * R

		return T

class NUFFTAdjoint(Linop):
	def __init__(self, oshape, coord, oversamp=1.25, width=4):
		self.coord = coord
		self.oversamp = oversamp
		self.width = width

		ndim = coord.shape[-1]

		ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])

		super().__init__(oshape, ishape)


	def _apply(self, input):
		return sf.nufft_adjoint(
			input, self.coord, self.oshape,
			oversamp=self.oversamp, width=self.width)

	def _adjoint_linop(self):
		return NUFFT(self.oshape, self.coord,
					 oversamp=self.oversamp, width=self.width)



