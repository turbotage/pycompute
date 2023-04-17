import numpy as np
import scipy as sp
import scipy.fftpack as spfft

import matplotlib.pyplot as plt

N1 = 64
shape = (N1,)
shape_2n = tuple([2*x for x in shape])

a = (np.random.rand(*shape) + 1j*np.random.rand(*shape)).astype(np.complex64)
a = np.random.rand(*shape).astype(np.float32)

dct1 = spfft.dct(a)

dct2 = a
dct2 = np.fft.fft(dct2, 2*N1)
dct2 = dct2[:shape[0]]
dct2 *= 2*np.exp(-1j*np.pi*np.arange(N1) / (2*N1))
dct2 = np.real(dct2)



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


