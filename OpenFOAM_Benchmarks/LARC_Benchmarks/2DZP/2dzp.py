#
import fluidfoam as ff
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

U  = 69.4
nu = 1.388e-05
bc = 'bottomWall'
xplt = 0.97

nplt = 26
xPlot = xplt * np.ones(nplt)
zPlot = 1e-8 * 2**np.arange(nplt)

tstep = 'latestTime'

def zero_pres(casedir, nx, nz):

    ## skin friction plot
    xWall  = ff.readfield(casedir, time_name=tstep, name='Cx', boundary=bc)
    txWall = ff.readfield(casedir, time_name=tstep, name='wallShearStress', \
                          boundary=bc)[0]
    cfWall = np.abs(txWall) / (0.5*U*U)

    ## uplus-yplus plot
    #x, y, z  = ff.readmesh(casedir)
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
    
    ufWall = np.sqrt(np.abs(txWall))
    ufInterp = interp.interp1d(xWall, ufWall, kind='cubic')
    ufPlot = ufInterp(xplt)
    
    uPlus = uPlot / ufPlot
    yPlus = zPlot / (nu / ufPlot)
    yPlus[0] = 1
    logYP = np.log10(yPlus)

    return xWall, cfWall, logYP, uPlus
#
casedir = './challenge/2DZP/'
modeldir = './model2/'

nx = 136; nz = 6
#xw_c, cf_c, logYP_c, up_c = zero_pres(casedir + 'sa', nx, nz)
xw_l, cf_l, logYP_l, up_l = zero_pres(casedir + 'ml', nx, nz)

nx = 544; nz = 384
xw_d, cf_d, logYP_d, up_d = zero_pres(casedir + 'hf', nx, nz)

fig, ax = plt.subplots(2, 1)
plt.suptitle('2DZP')

# coef. of friction
ax[0].plot(xw_d, cf_d, '-', color='k', label='dense')
# ax[0].plot(xw_c, cf_c, ':', color='k', label='coarse')
ax[0].plot(xw_l, cf_l, '--', color='k', label='ml')
ax[0].set_ylim(0.002, 0.006)
ax[0].legend(loc='upper right')
ax[0].set_xlabel('x')
ax[0].set_ylabel(f'$C_f$')

# uplus, yplus
ax[1].plot(logYP_d, up_d, '-', color='k', label='dense')
# ax[1].plot(logYP_c, up_c, ':', color='k', label='coarse')
ax[1].plot(logYP_l, up_l, '--', color='k', label='ml')
ax[1].set_xlim(0.00, 4)
ax[1].set_ylim(0.00, 30)
ax[1].legend(loc='lower right')
ax[1].set_xlabel(f'$log(y+)$')
ax[1].set_ylabel(f'$u+$')


fig.tight_layout()
plt.savefig(modeldir + '2dzp.png')
plt.show(block=False)

cf = {
      'x': xw_l,
      'cf': cf_l
     }
cf = pd.DataFrame(cf)
cf.to_csv(modeldir + '2dzp_cf.dat', sep='\t')

upyp = {
        'log10Yplus': logYP_l,
        'uplus': up_l
       }
upyp = pd.DataFrame(upyp)
upyp.to_csv(modeldir + '2dzp_upyp.dat', sep='\t')
#
