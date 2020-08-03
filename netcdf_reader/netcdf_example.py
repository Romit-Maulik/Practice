'''
REFERENCES
    netcdf4-python -- http://code.google.com/p/netcdf4-python/
    NCEP/NCAR Reanalysis -- Kalnay et al. 1996
        http://dx.doi.org/10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2
    https://iescoders.com/reading-netcdf4-data-in-python/ - useful tutorial
'''
import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt


if __name__ == '__main__':
    download_data = False
    print_info = False

    nc_f = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.oisst.v2/sst.wkmean.1981-1989.nc?time[0:426],sst[0:426][0:179][0:359]'  # Your filename
    nc_f_new = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.oisst.v2/sst.wkmean.1990-present.nc?time[0:1486],sst[0:1486][0:179][0:359]'
    source_mask = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.oisst.v2/lsmask.nc?lat[0:1:179],lon[0:1:359],mask[0][0:179][0:359]';

    if download_data:
        nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                                     # and create an instance of the ncCDF4 class

        if print_info:
            print(nc_fid)
            # Dimensions
            for d in nc_fid.dimensions.items():
                print(d)

            for d in nc_fid.variables.items():
                print(d)

        mask_f = Dataset(source_mask,'r')

        if print_info:
            print(mask_f)

        # # Variables
        sst = nc_fid.variables["sst"][:]
        time_var = nc_fid.variables["time"][:]
        mask = mask_f.variables["mask"][:]
        lat = mask_f.variables["lat"][:]
        lon = mask_f.variables["lon"][:]

        sst.dump('sst_var')
        time_var.dump('time_var')
        mask.dump('mask')
        lat.dump('lat')
        lon.dump('lon')

    else:
        sst = np.load('sst_var',allow_pickle=True)
        time_var = np.load('time_var',allow_pickle=True)
        lat = np.load('lat',allow_pickle=True)
        lon = np.load('lon',allow_pickle=True)
        mask = np.load('mask',allow_pickle=True)

        print(np.shape(lat),np.shape(lon),np.shape(sst),np.shape(mask))

        sst = sst*mask

        for t in range(10):
            plt.figure()
            cs = plt.contourf(lon,lat,sst[t,:,:])
            plt.colorbar()
            plt.show()




    