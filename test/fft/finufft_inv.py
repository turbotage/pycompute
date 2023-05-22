import numpy as np
import matplotlib.pyplot as plt

import finufft

N1 = 4
N2 = 4
N3 = 4
NX = N1*N2*N3
NF = 2*NX

coord = np.empty((3,NF), dtype=np.float32)
l = 0
for x in range(N1):
	for y in range(N2):
		for z in range(N3):
			kx = -np.pi + x * 2 * np.pi / N1
			ky = -np.pi + y * 2 * np.pi / N2
			kz = -np.pi + z * 2 * np.pi / N3

			coord[0,l] = kx
			coord[1,l] = ky
			coord[2,l] = kz

			l += 1
			
#coord = np.transpose(coord, axes=(1,0))
coord = np.random.rand(NF,3).astype(np.float32) #(np.random.rand(NF,3) + 1j*np.random.rand(NF,3)).astype(np.complex64)

inputimg = (np.random.rand(N1,N2,N3) + 1j*np.random.rand(N1,N2,N3)).astype(np.complex64)

transformed = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], inputimg) / np.sqrt(N1*N2*N3)
transformed = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], transformed, (N1,N2,N3)) / np.sqrt(N1*N2*N3)

plt.figure()
plt.plot(np.abs(inputimg.flatten()), 'r-')
plt.plot(np.abs(transformed.flatten()), 'g-')
plt.legend(['input', 'transformed'])
plt.show()