import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from testrun import test_runner
test_runner()

import h5py
import pycompute.cuda.resample as res

filemap = 'D:\\4D-Recon\\Viktor\\'
filename_images = filemap + 'images.h5'
filename_sense = filemap + 'SenseMapsCpp.h5'
filename_coords = filemap + 'MRI_Raw_resampled.h5'


datas = res.full_resampling(filename_images, filename_sense, filename_coords, [100,100,100], 2)

print('Ello')


