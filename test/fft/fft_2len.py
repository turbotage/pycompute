import time
import numpy as np
import cupy as cp

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NX = 32

p = (cp.random.uniform(0,1,(NX,)) + cp.random.uniform(0,1,(NX,))).astype(cp.complex64)

pmod = p.copy()
#pmod[NX // 2] = 0.5*(p[NX // 2 - 1] + p[NX // 2 + 1])
pmod[NX // 2] = pmod[0]

ip = cp.fft.ifft(p)
ipmod = cp.fft.ifft(pmod)


plt.figure()
plt.plot(p.get())
plt.plot(pmod.get())
plt.legend(['p', 'pmod'])
plt.show()


plt.figure()
plt.plot(ip.get())
plt.plot(ipmod.get())
plt.legend(['ip', 'ipmod'])
plt.show()

