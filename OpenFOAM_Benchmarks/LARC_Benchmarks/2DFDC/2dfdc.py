#
import fluidfoam as ff
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import pandas as pd

U  = 8.0
nu = 1.0e-7
bc = 'bottom'
xplt = 500

nplt = 26
xPlot = xplt * np.ones(nplt)
zPlot = 1e-8 * 2**np.arange(nplt)

tstep = 'latestTime'

def channel(casedir, nx, nz):

    # friction vel
    xWall  = ff.readfield(casedir, time_name=tstep, name='Cx', boundary=bc)
    txWall = ff.readfield(casedir, time_name=tstep, name='wallShearStress', \
                          boundary=bc)[0]
    ufWall = np.sqrt(np.abs(txWall))
    ufInterp = interp.interp1d(xWall, ufWall, kind='cubic')
    ufPlot = ufInterp(xplt)

    cfPlot = ufPlot * ufPlot / (0.5*U*U)
    print('skin friction at x=', xplt, ': ', cfPlot)
    
    ## uplus-yplus plot
    x = ff.readfield(casedir, time_name=tstep, name='Cx')
    z = ff.readfield(casedir, time_name=tstep, name='Cz')
    u = ff.readfield(casedir, time_name=tstep, name='U')[0]
    
    xx = np.reshape(x, (nx, nz))
    zz = np.reshape(z, (nx, nz))
    uu = np.reshape(u, (nx, nz))
    
    xz = np.zeros((x.size, 2))
    xz[:,0] = x
    xz[:,1] = z
    uPlot = interp.griddata(xz, u, (xPlot, zPlot), method='linear')
    uPlot[0] = 0
    
    uPlus = uPlot / ufPlot
    yPlus = zPlot / (nu / ufPlot)
    yPlus[0] = 1
    logYP = np.log10(yPlus)
    
    return logYP, uPlus

casedir = './challenge/2DFDC/'
modeldir = './model2/'

nx = 10; nz = 32
#logYP_c, up_c = channel(casedir + 'sa', nx, nz)
logYP_l, up_l = channel(casedir + 'ml', nx, nz)

nx = 160; nz = 512
logYP_d, up_d = channel(casedir + 'hf', nx, nz)

fig, ax = plt.subplots()
plt.suptitle('2DFDC')

# uplus, yplus
ax.plot(logYP_d, up_d, '-', color='k', label='dense')
ax.plot(logYP_l, up_l, '--', color='k', label='ml')
ax.legend(loc='lower right')
ax.set_xlabel(f'$log(y+)$')
ax.set_ylabel(f'$u+$')

fig.tight_layout()
plt.savefig(modeldir + '2dfdc.png')
plt.show(block=False)

upyp = {
        'log10Yplus': logYP_l,
        'uplus': up_l
       }
upyp = pd.DataFrame(upyp)
upyp.to_csv(modeldir + '2dfdc_upyp.dat', sep='\t')
#
