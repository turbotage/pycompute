# imports
from matplotlib import pyplot as plt
import numpy as np
import math
import time

import importlib

import cupy as cp

from mpl_toolkits.mplot3d import Axes3D

from scipy import signal, misc

from dipy.data import get_fnames
from dipy.io.image import load_nifti_data, save_nifti, load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

print('After imports')

fraw,fbval,fbvec = get_fnames('ivim')

_, affine = load_nifti(fraw)
data = np.float32(load_nifti_data(fraw))[:,:,15:35,:]
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs, b0_threshold=0)

data_shape = data.shape
nvoxels = data_shape[0]*data_shape[1]*data_shape[2]
ndata = data_shape[3]
Nelem = nvoxels*ndata

data_flat = np.float32(data.reshape(1, Nelem))
consts = np.float32(np.reshape(np.tile(bvals, nvoxels), (1,Nelem)))

redo_pars_t = False
if redo_pars_t:

	from dipy.reconst.ivim import IvimModel
	ivimmodel = IvimModel(gtab, fit_method='trr')

	ivimfit = ivimmodel.fit(data)

	pars_t = ivimfit.model_params

	save_nifti('pars_dipy.nii', pars_t, affine)

pars_dipy, affine = load_nifti('pars_dipy.nii')
pars_dipy = np.float32(pars_dipy)[:,:,15:35,:]

lower_bound = cp.empty((4, nvoxels), dtype=np.float32)
lower_bound[0,:] = np.finfo(np.float32).min
lower_bound[1,:] = 0.0
lower_bound[2,:] = 0.0
lower_bound[3,:] = 0.0

upper_bound = cp.empty((4, nvoxels), dtype=np.float32)
upper_bound[0,:] = np.finfo(np.float32).max / 2
upper_bound[1,:] = 1.0
upper_bound[2,:] = 1.0
upper_bound[3,:] = 1.0

print('After DIPY loads')

kernel_size = 5
avg_kernel = np.ones((kernel_size,kernel_size))

pars = np.empty(pars_dipy.shape, pars_dipy.dtype)

for i in range(0,pars.shape[2]):
	for j in range(0,4):
		pars[:,:,i,j] = signal.convolve2d(pars_dipy[:,:,i,j], avg_kernel, boundary='symm', mode='same')

print('After convolution')

def param_printer(pars, slicez=15, clim=[0.0, 16000], viewport=[[0.0, 1.0], [0.0, 1.0]], print_S0=False, print_f=False, print_D1=False, print_D2=False):
	xlen = pars.shape[0]
	ylen = pars.shape[1]
	xstart = round(xlen*viewport[0][0])
	xend = round(xlen*viewport[0][1])
	ystart = round(ylen*viewport[1][0])
	yend = round(ylen*viewport[1][1])

	if print_S0:
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0,0,1,1])
		figdata = ax1.imshow(pars[xstart:xend,ystart:yend,slicez,0])
		ax1.set_title('S0')
		figdata.set_clim(clim[0], clim[1])
		fig1.colorbar(figdata, ax=ax1)
		plt.show()

	if print_f:
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0,0,1,1])
		figdata = ax1.imshow(pars[xstart:xend,ystart:yend,slicez,1])
		ax1.set_title('f')
		figdata.set_clim(clim[0], clim[1])
		fig1.colorbar(figdata, ax=ax1)
		plt.show()

	if print_D1:
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0,0,1,1])
		figdata = ax1.imshow(pars[xstart:xend,ystart:yend,slicez,2])
		ax1.set_title('D1')
		figdata.set_clim(clim[0], clim[1])
		fig1.colorbar(figdata, ax=ax1)
		plt.show()

	if print_D2:
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0,0,1,1])
		figdata = ax1.imshow(pars[xstart:xend,ystart:yend,slicez,3])
		ax1.set_title('D2')
		figdata.set_clim(clim[0], clim[1])
		fig1.colorbar(figdata, ax=ax1)
		plt.show()

	
pars_flat = np.reshape(np.transpose(pars, (3,0,1,2)), (4, nvoxels)).copy()

import cuda.lsqnonlin as clsq
importlib.reload(clsq)

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))'
pars_str = ['S0', 'f', 'D_1', 'D_2']
consts_str = ['b']

nchunks = 4
chunk_size = math.ceil(nvoxels / nchunks)

data_flat[:,0:21] = np.reshape(np.array([908.02686, 905.39154, 906.08997, 700.7829, 753.0848, 859.9136,
	   870.48846, 755.96893, 617.3499, 566.2044 , 746.62067, 662.47424,
	   628.8806, 459.7746 , 643.30554, 318.58453, 416.5493, 348.34335,
	   411.74026, 284.17468, 290.30487]), (1,21))

pars_flat[:,0] = np.array([700.0, 0.2, 0.1, 0.001])

solm = clsq.SecondOrderLevenbergMarquardt(expr, pars_str, consts_str, ndata=21, dtype=cp.float32, write_to_file=True)

start = time.time()
for i in range(0,nchunks):

	parscu = cp.array(pars_flat[:,i*chunk_size:(i+1)*chunk_size], dtype=cp.float32, copy=True)
	constscu = cp.array(consts[:,i*chunk_size*ndata:(i+1)*chunk_size*ndata], dtype=cp.float32, copy=True)
	datacu = cp.array(data_flat[:,i*chunk_size*ndata:(i+1)*chunk_size*ndata], dtype=cp.float32, copy=True)
	lower_bound_cu = cp.array(lower_bound[:,i*chunk_size:(i+1)*chunk_size], dtype=cp.float32, copy=True)
	upper_bound_cu = cp.array(upper_bound[:,i*chunk_size:(i+1)*chunk_size], dtype=cp.float32, copy=True)

	solm.setup(parscu, constscu, datacu, lower_bound_cu, upper_bound_cu)
	solm.run(30, 1e-5)
	
	pars_flat[:,i*chunk_size:(i+1)*chunk_size] = parscu.get()

	#time.sleep(5)

cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print('It took: ' + str(end - start) + ' s')

del parscu
del constscu
del datacu
del lower_bound_cu
del upper_bound_cu

pars_flat_back = np.transpose(np.reshape(pars_flat, (4, data_shape[0], data_shape[1], data_shape[2])), (1,2,3,0))

#param_printer(pars_dipy, print_D1=True)

param_printer(pars_dipy, slicez=15, clim=[0.0, 25000], viewport=[[0.0, 1.0],[0.0, 1.0]], print_S0=True)

param_printer(pars_flat_back, slicez=15, clim=[0.0, 25000], viewport=[[0.0, 1.0],[0.0, 1.0]], print_S0=True)

input("Press Enter...")
