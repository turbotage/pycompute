from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def param_printer(pars, slicez=15, clim=[0.0, 16000], viewport=[[0.0, 1.0], [0.0, 1.0]], pars_idx: list[int] = [0]):
	xlen = pars.shape[0]
	ylen = pars.shape[1]
	xstart = round(xlen*viewport[0][0])
	xend = round(xlen*viewport[0][1])
	ystart = round(ylen*viewport[1][0])
	yend = round(ylen*viewport[1][1])

	for i in pars_idx:
		fig1 = plt.figure()
		ax1 = fig1.add_axes([0,0,1,1])
		figdata = ax1.imshow(pars[xstart:xend,ystart:yend,slicez,i])
		ax1.set_title("I'th parameter")
		figdata.set_clim(clim[0], clim[1])
		fig1.colorbar(figdata, ax=ax1)



