#
import fluidfoam as ff
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

alpha = np.array([10, 15, 17, 18])

U  = 6.00
nu = 1.0e-06
bc = 'walls'
tstep = 'latestTime'

def airfoil(casedir):

    xWall  = ff.readfield(casedir, time_name=tstep, name='Cx', boundary=bc)
    zWall  = ff.readfield(casedir, time_name=tstep, name='Cz', boundary=bc)
    cpWall = ff.readfield(casedir, time_name=tstep, name='Cp', boundary=bc)
    txWall, tyWall, tzWall = ff.readfield(casedir, time_name=tstep, \
                                          name='wallShearStress', boundary=bc)

    #cfWall = np.sqrt(txWall**2 + tyWall**2 + tzWall**2) / (0.5*U*U)
    cfWall = np.abs(txWall) / (0.5*U*U)

    upper = zWall >= 0
    upper = zWall <= 0
    xx = xWall  * upper
    cf = cfWall * upper

    return xWall, cpWall, xx, cf
#

casedir = './challenge/2DN00_alpha00/'

xw_c, cp_c, xx_c, cf_c =  airfoil(casedir + 'sa') # coarse
xw_d, cp_d, xx_d, cf_d =  airfoil(casedir + 'hf') # dense

plt.close('all')

#ax = subplot(1,1,1)
# pressure coeff - upper and lower surface
plt.figure()
plt.title('Cp')
plt.plot(xw_c, cp_c, '*-', label='coarse')
plt.plot(xw_d, cp_d, '-', label='dense')
plt.ylim(-1, 1.5)
plt.legend(loc='lower right')
plt.show(block=False)

# friction coeff - upper surface only
plt.figure()
plt.title('Cf')
plt.plot(xx_c, cf_c, '*-', label='coarse')
plt.plot(xx_d, cf_d, '-', label='dense')
plt.ylim(-0.05, 0.1)
plt.legend(loc='lower right')
plt.show(block=False)
#
