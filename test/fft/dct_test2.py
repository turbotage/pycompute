import numpy as np
import scipy as sp
import scipy.fftpack as spfft

import matplotlib.pyplot as plt

N1 = 4
N2 = 7
N3 = 6

shape = (N1,N2,N3)
shape_2n = tuple([2*x for x in shape])

a = (np.random.rand(*shape) + 1j*np.random.rand(*shape)).astype(np.complex64)
a = np.random.rand(*shape).astype(np.float32)

dct1 = spfft.dctn(a)

def dct(x, axis, aranged_exp = None):
	len = x.shape[axis]
	if aranged_exp == None:
		aranged_exp = 2*np.exp(-0.5j*np.pi*np.arange(len) / len)

	y = np.fft.fft(x, 2*len, axis).take(indices=range(0,len), axis=axis)
	if axis==0:
		y *= aranged_exp[:,np.newaxis,np.newaxis]
	elif axis==1:
		y *= aranged_exp[np.newaxis,:,np.newaxis]
	elif axis==2:
		y *= aranged_exp[np.newaxis,np.newaxis,:]
	else:
		raise RuntimeError('Unsupported axis in dct')
	return np.real(y)

def dct3(x):
	y = dct(x, axis=0)
	y = dct(y, axis=1)
	y = dct(y, axis=2)
	return y

dct2 = dct3(a)

dct1 = dct1.flatten()
dct2 = dct2.flatten()


plt.figure()
plt.plot(np.abs(dct1))
plt.plot(np.abs(dct2))
plt.show()

plt.figure()
plt.plot(np.real(dct1))
plt.plot(np.real(dct2))
plt.show()

plt.figure()
plt.plot(np.imag(dct1))
plt.plot(np.imag(dct2))
plt.show()


