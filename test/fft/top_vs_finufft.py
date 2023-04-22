
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from testrun import test_runner
test_runner()

import numpy as np
import cupy as cp

import pycompute.cuda.sigpy.fourier_linops as fulinops
import pycompute.cuda.sigpy.fourier as fourier
import pycompute.cuda.sigpy.linop as linop

import matplotlib.pyplot as plt

import finufft

def print_array(arr):
	for z in range(arr.shape[2]):
		print(" [")
		for y in range(arr.shape[1]):
			str1 = " ["
			for x in range(arr.shape[0]):
				val = arr[x,y,z].item()
				if val >= 0.0:
					str1 += ' '
				str1 += '{:.2e}'.format(val)
				str1 += ","
			print(str1, "]")
		print(" ]")
	print("]")


N1 = 2
N2 = 2
N3 = 1
NX = N1*N2*N3
NF = NX

large = (2*N1,2*N2,2*N3)
small = (N1,N2,N3)

coord = np.empty((3,NF), dtype=np.float32)
l = 0
for x in range(N1):
	for y in range(N2):
		for z in range(N3):
			kx = -np.pi + x * 2.0 * np.pi / N1
			ky = -np.pi + y * 2.0 * np.pi / N2
			kz = -np.pi + z * 2.0 * np.pi / N3

			coord[0,l] = kx #+ 0.01
			coord[1,l] = ky #+ 0.01
			coord[2,l] = kz #+ 0.01

			l += 1

#coord[0,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32)
#coord[1,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32)
#coord[2,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32)

# print(coord)
# plt.figure()
# plt.plot(coord[0,:], 'r-')
# plt.plot(coord[1,:], 'g-')
# plt.plot(coord[2,:], 'b-')
# plt.legend(['x', 'y', 'z'])
# plt.show()

	
fixed_coords = np.array([
	[0.98032004, 0.34147135, 0.61084439],
	[0.80602593, 0.65668227, 0.58148698],
	[0.11397716, 0.33431708, 0.98374713],
	[0.49615291, 0.28232985, 0.99452616],
	[0.6371665 , 0.17878657, 0.60917358],
	[0.12547691, 0.80906753, 0.93934428],
	[0.27922639, 0.34255417, 0.93016733],
	[0.5219615 , 0.09645867, 0.03214259],
	[0.27719558, 0.53159027, 0.17859567],
	[0.46942625, 0.34165946, 0.76490134],
	[0.75368101, 0.19968615, 0.28943047],
	[0.54910582, 0.42412554, 0.53432473],
	[0.87679761, 0.50310224, 0.97346862],
	[0.90748122, 0.19441631, 0.28978457],
	[0.88753202, 0.82203755, 0.9381962 ],
	[0.4025496 , 0.1600768 , 0.57524011],
	[0.65525699, 0.30591684, 0.61524885],
	[0.25528118, 0.40408854, 0.41270846],
	[0.83251984, 0.65801961, 0.59457825],
	[0.03506295, 0.9506098 , 0.30317166],
	[0.93298131, 0.89051245, 0.26771778],
	[0.98649964, 0.86362155, 0.78970084],
	[0.63909318, 0.1392839 , 0.86418488],
	[0.54289101, 0.63570404, 0.90898353],
	[0.82511889, 0.42676573, 0.32190359],
	[0.31232441, 0.97744591, 0.15726858],
	[0.97187048, 0.24543457, 0.63135005],
	[0.79675513, 0.36351348, 0.01525023],
	[0.82062195, 0.03481492, 0.59195562],
	[0.4450689 , 0.8571279 , 0.96007808],
	[0.9264222 , 0.69303613, 0.10664806],
	[0.71865813, 0.54588828, 0.78308924],
	[0.68904944, 0.0625562 , 0.61408031],
	[0.14991815, 0.97537933, 0.53627866],
	[0.12675083, 0.9740082 , 0.9854024 ],
	[0.97176999, 0.8914261 , 0.13173493],
	[0.1824385 , 0.11983747, 0.38931872],
	[0.54571302, 0.17167634, 0.1974539 ],
	[0.6698734 , 0.03353057, 0.52053646],
	[0.38529699, 0.56414308, 0.84647764],
	[0.2090367 , 0.93103123, 0.4905254 ],
	[0.67817081, 0.9677812 , 0.6214241 ],
	[0.32259623, 0.62142206, 0.82752577],
	[0.44352566, 0.58230826, 0.253879  ],
	[0.5019813 , 0.7628638 , 0.99163866],
	[0.43278585, 0.04287191, 0.57724611],
	[0.37305061, 0.85525078, 0.79450935],
	[0.54032247, 0.45156145, 0.51494839],
	[0.43547165, 0.47884727, 0.83920057],
	[0.90771454, 0.44304607, 0.10684609],
	[0.96914703, 0.99099193, 0.67660616],
	[0.12878615, 0.87274706, 0.63536308],
	[0.08763589, 0.96869032, 0.78334944],
	[0.89292773, 0.17549878, 0.63936669],
	[0.04983882, 0.84603837, 0.01729135],
	[0.28048693, 0.26340963, 0.37187648],
	[0.00647681, 0.65307172, 0.36669731],
	[0.51984174, 0.65083299, 0.22426562],
	[0.76060238, 0.54593353, 0.90458955],
	[0.06881982, 0.55241897, 0.20386456],
	[0.40423935, 0.19247943, 0.01989418],
	[0.51691671, 0.3127277 , 0.59669834],
	[0.88841294, 0.98875191, 0.02759931],
	[0.33364495, 0.87896414, 0.66459484],
	[0.89172249, 0.98091925, 0.93169932],
	[0.23032556, 0.63535047, 0.85921909],
	[0.0318257 , 0.5379664 , 0.82014012],
	[0.10007126, 0.42387433, 0.57601986],
	[0.22774557, 0.36218039, 0.17861717],
	[0.03882169, 0.56421647, 0.56901133],
	[0.89507728, 0.11861613, 0.24141007],
	[0.99859803, 0.68317083, 0.09806365],
	[0.71303893, 0.73025946, 0.78945573],
	[0.1102953 , 0.62754217, 0.73104685],
	[0.4376749 , 0.77870555, 0.62067626],
	[0.41754434, 0.85778273, 0.06900474],
	[0.66257871, 0.09020336, 0.30395005],
	[0.46025083, 0.21331427, 0.52045147],
	[0.18048847, 0.67956526, 0.94928918],
	[0.37032696, 0.58974144, 0.31576817],
	[0.06239387, 0.22940668, 0.62886661],
	[0.25989873, 0.43018133, 0.34109312],
	[0.29890908, 0.62653838, 0.15048865],
	[0.26687828, 0.69478892, 0.35177644],
	[0.6200615 , 0.59579733, 0.02966846],
	[0.40749241, 0.28749332, 0.12400579],
	[0.2824288 , 0.68365401, 0.67101405],
	[0.10994612, 0.75164738, 0.32341936],
	[0.61737234, 0.02443164, 0.67057934],
	[0.02651275, 0.23172328, 0.06784856],
	[0.70089601, 0.88461766, 0.66256881],
	[0.5713635 , 0.62115033, 0.47254448],
	[0.13134128, 0.27217286, 0.52773285],
	[0.45412968, 0.94835191, 0.0894173 ],
	[0.08491943, 0.39140606, 0.24737922],
	[0.21292069, 0.60481292, 0.18213903],
	[0.6853412 , 0.13670174, 0.5994464 ],
	[0.08583449, 0.31034405, 0.02131659],
	[0.77788847, 0.15017276, 0.35985071],
	[0.13258452, 0.5643354 , 0.79174118]
]).astype(np.float32)

fixed_coords = -np.pi + 2 * np.pi * fixed_coords

for i in range(NF):
	coord[0,i] = fixed_coords[i,0]
	coord[1,i] = fixed_coords[i,1]
	coord[2,i] = fixed_coords[i,2]

coord = np.transpose(coord, (1,0))
#c = np.ones((1,N1,N2,N3), dtype=cp.complex64)

sp_coords = cp.array(coord)
sp_coords[:,0] *= (N1 / (2*cp.pi))
sp_coords[:,1] *= (N2 / (2*cp.pi))
sp_coords[:,2] *= (N3 / (2*cp.pi))
#sp_temp = sp_coords[:,0]
#sp_coords[:,0] = sp_coords[:,2]
#sp_coords[:,2] = sp_temp

psf1 = fourier.toeplitz_psf(sp_coords, (1,N1,N2,N3), oversamp=2.0, width=4).get()
#psf1 = np.fft.ifftn(psf1)
psf1_f = psf1.flatten()

unity_vector = np.zeros((2*N1,2*N2,2*N3), dtype=cp.complex64)
#unity_vector[0,0,0] = 1
unity_vector[N1,N2,N3] = 1.0
#print('unity_vector: ', unity_vector)
psf2 = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], unity_vector)
psf2 = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], psf2, (2*N1,2*N2,2*N3))

print('coord: ', coord)

#print('diagonal real:\n', np.real(psf2.swapaxes(0,2)))
#print('diagonal imag:\n', np.imag(psf2.swapaxes(0,2)))

print('real')
print_array(np.real(psf2))
print('imag')
print_array(np.imag(psf2))

psf2 = np.fft.ifftshift(psf2)
psf2 = np.fft.fftn(psf2)

#print('diagonal real:\n', np.real(psf2.transpose((2,1,0))))
#print('diagonal imag:\n', np.imag(psf2.transpose((2,1,0))))

psf2_f = psf2.flatten()

plt.figure()
plt.plot(np.abs(psf1_f), 'r-')
plt.plot(np.abs(psf2_f), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.real(psf1_f), 'r-')
plt.plot(np.real(psf2_f), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.imag(psf1_f), 'r-')
plt.plot(np.imag(psf2_f), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

p = (np.random.rand(1,N1,N2,N3).astype(np.float32) + 1j*np.random.rand(1,N1,N2,N3).astype(np.float32)).astype(np.complex64)
#p = np.arange(N1*N2*N3).astype(np.complex64).reshape(1,N1,N2,N3)
#p = 0.5 * np.ones((1,N1,N2,N3), dtype=np.complex64) + np.exp(2*np.pi*p) + np.cos(4*np.pi*p)
p = np.ones((1,N1,N2,N3), dtype=np.complex64)
# plt.figure()
# plt.plot(np.real(p.flatten()), 'r-')
# plt.plot(np.imag(p.flatten()), 'g-')
# plt.legend(['p real', 'p imag'])
# plt.show()

nufft_out1 = p
sp_nufft1_out = False
if sp_nufft1_out:
	#normal_nufft = fulinops.NormalNUFFT((1,N1,N2,N3), sp_coords, oversamp=2.0, width=4)
	#nufft_out1 = normal_nufft.apply(cp.array(nufft_out1)).get()
	nufft_out1 = fourier.nufft(cp.array(nufft_out1), sp_coords, oversamp=2.0, width=4)
	nufft_out1 = fourier.nufft_adjoint(nufft_out1, sp_coords, (1,N1,N2,N3), oversamp=2.0, width=4).get()
	#nufft_out1 = np.fft.fftshift(nufft_out1)
else:
	nufft_out1 = nufft_out1.squeeze(0)
	nufft_out1 = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], nufft_out1)
	nufft_out1 = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], nufft_out1, (N1,N2,N3))
	#nufft_out1 = np.fft.fftshift(nufft_out1)

diag = psf2
diag = diag
#diag = np.fft.ifftshift(diag)

nufft_out2 = p
nufft_out2 = np.fft.fftn(nufft_out2, s=(2*N1,2*N2,2*N3), axes=(1,2,3))
nufft_out2 = diag * nufft_out2
nufft_out2 = np.fft.ifftn(nufft_out2, s=(2*N1,2*N2,2*N3), axes=(1,2,3))
#nufft_out2 = np.fft.ifftn(nufft_out2, s=(N1,N2,N3), axes=(1,2,3))
#nufft_out2 = nufft_out2[:,(N1 // 2):((N1 // 2) + N1),(N2 // 2):((N2 // 2) + N2),(N3 // 2):((N3 // 2) + N3)]
nufft_out2 = nufft_out2[:,:N1,:N2,:N3]

#nufft_out2 = np.fft.ifftshift(nufft_out2)

#nufft_out1 = np.fft.ifftshift(nufft_out1)
#nufft_out2 = np.fft.ifftshift(nufft_out2)

nufft_out1 = nufft_out1.flatten()
nufft_out2 = nufft_out2.flatten()
#nufft_out1 = nufft_out1[:100]
#nufft_out2 = nufft_out2[:100]

plt.figure()
plt.plot(np.abs(nufft_out1), 'r-')
plt.plot(np.abs(nufft_out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.real(nufft_out1), 'r-')
plt.plot(np.real(nufft_out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()

plt.figure()
plt.plot(np.imag(nufft_out1), 'r-')
plt.plot(np.imag(nufft_out2), 'g-')
plt.legend(['nufft1', 'nufft2'])
plt.show()


#print(np.real(nufft_out1).flatten())
#print(np.imag(nufft_out1).flatten())
#print(np.abs(nufft_out1).flatten())