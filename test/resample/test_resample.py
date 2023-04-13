
import cupy as cp

import numpy as np
import math

import h5py

def copy_to_mean_h5(filename: str, n: int):
    filename_out = filename[:-3] + '_out.h5'
    with h5py.File(filename, "r") as f:
        
        print("Keys: %s" % f.keys())

        kx = np.zeros_like(f['Kdata']['KX_E0'][()])
        ky = np.zeros_like(f['Kdata']['KY_E0'][()])
        kz = np.zeros_like(f['Kdata']['KZ_E0'][()])
        for i in range(5):
            kx += f['Kdata']['KX_E' + str(i)][()]
            ky += f['Kdata']['KY_E' + str(i)][()]
            kz += f['Kdata']['KZ_E' + str(i)][()]
        kx /= 5
        ky /= 5
        kz /= 5

        with h5py.File(filename_out, "w") as o:
            f.copy(f['Kdata'], o, 'Kdata')
            f.copy(f['Gating'], o, 'Gating')

            for i in range(5):
                o['Kdata']['KX_E'+str(i)][()] = kx
                o['Kdata']['KY_E'+str(i)][()] = ky
                o['Kdata']['KZ_E'+str(i)][()] = kz

def resample_coords_from_h5(filename: str, filename_out: str, n: int):
    with h5py.File(filename, "r") as f:
        with h5py.File(filename_out, "w") as o:

            o.create_group('Kdata')
            o.create_group('Gating')

            encs=5
            coils=32
            for i in range(coils):
                for j in range(encs):
                    dn = 'KData_E' + str(j) + '_C' + str(i)

                    kdata = f['Kdata'][dn][()][0,::(n*n),::n]

                    o['Kdata'].create_dataset(dn, data=kdata)

            for i in range(encs):
                tn = 'KT_E' + str(i)
                kwn = 'KW_E' + str(i)
                kxn = 'KX_E' + str(i)
                kyn = 'KY_E' + str(i)
                kzn = 'KZ_E' + str(i)

                datat = f['Kdata'][tn][()][0,::(n*n),::n]
                dataw = f['Kdata'][kwn][()][0,::(n*n),::n]
                datax = f['Kdata'][kxn][()][0,::(n*n),::n]
                datay = f['Kdata'][kyn][()][0,::(n*n),::n]
                dataz = f['Kdata'][kzn][()][0,::(n*n),::n]

                o['Kdata'].create_dataset(tn, data=datat)
                o['Kdata'].create_dataset(kwn, data=dataw)
                o['Kdata'].create_dataset(kxn, data=datax)
                o['Kdata'].create_dataset(kyn, data=datay)
                o['Kdata'].create_dataset(kzn, data=dataz)


if __name__ == '__main__':
    resample_coords_from_h5('D:\\4D-Recon\\MRI_Raw.h5', 'D:\\4D-Recon\\MRI_Raw_resampled.h5', 2)




    


