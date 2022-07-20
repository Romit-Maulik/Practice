#
import fluidfoam as ff
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

U  = 9.29
nu = 1.0e-05
bc = 'walls'
xplt = np.array([0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
zplt = np.array([0.1162, 0.0245, 0.0048, 0, 0, 0, 0])

nplt = 101
xPlot = xplt.reshape(7, 1) * np.ones((1,nplt))

zPlot = np.zeros((7,nplt))
for i in range(7):
    zPlot[i,:] = np.linspace(zplt[i], 0.16, nplt)

tstep = 'latestTime'

xwplt = np.linspace(-2,2,101)

def hump(casedir, nx, nz):

    ## friction / pressure coeff
    xWall  = ff.readfield(casedir, time_name=tstep, name='Cx', boundary=bc)
    txWall = ff.readfield(casedir, time_name=tstep, name='wallShearStress', \
                          boundary=bc)[0]

    cfWall = np.abs(txWall) / (0.5*U*U)
    cfInterp = interp.interp1d(xWall, cfWall, kind='cubic')
    cfPlot = cfInterp(xwplt)
    
    # pressure coeff
    cpWall = ff.readfield(casedir, time_name=tstep, name='Cp', boundary=bc)
    cpInterp = interp.interp1d(xWall, cpWall, kind='cubic')
    cpPlot = cpInterp(xwplt)
    
    # uplus-yplus plot
    x = ff.readfield(casedir, time_name=tstep, name='Cx')
    z = ff.readfield(casedir, time_name=tstep, name='Cz')
    u = ff.readfield(casedir, time_name=tstep, name='U')[0]

    # Reynolds stress tensor (xx, xy, xz, yy, yz, zz) 
    R = ff.readfield(casedir, time_name=tstep, name='turbulenceProperties:R')[2]

    xx = np.reshape(x, (nx, nz))
    zz = np.reshape(z, (nx, nz))

    xz = np.zeros((x.size, 2))
    xz[:,0] = x
    xz[:,1] = z
    uPlot = interp.griddata(xz, u, (xPlot, zPlot), method='linear') / U
    RPlot = interp.griddata(xz, R, (xPlot, zPlot), method='linear') / (U*U)

    return xWall, cfWall, cpWall, zPlot, uPlot, RPlot

#
casedir = './challenge/2DWMH/'
modeldir = './model2/'

#nx = 102; nz = 27
#xw_c, cf_c, cp_c, z_c, u_c, r_c = hump(casedir + 'sa_coarse', nx, nz)

nx = 204; nz = 54
#xw_c, cf_c, cp_c, z_c, u_c, r_c = hump(casedir + 'sa', nx, nz)
xw_l, cf_l, cp_l, z_l, u_l, r_l = hump(casedir + 'ml', nx, nz)

nx = 816; nz = 216
xw_d, cf_d, cp_d, z_d, u_d, r_d = hump(casedir + 'hf', nx, nz)

fig, ax = plt.subplots(4, 1, figsize=(8,20))
plt.suptitle('2DWMH')

# Cf
ax[0].plot(xw_d, cf_d, '-', color='k', label='dense')
# ax[0].plot(xw_c, cf_c, ':', color='k', label='coarse')
ax[0].plot(xw_l, cf_l, '--', color='k', label='ml')
ax[0].legend(loc='lower right')
ax[0].set_xlabel(f'$x/L$')
ax[0].set_ylabel(f'$C_f$')

# Cp
ax[1].plot(xw_d, cp_d, '-', color='k', label='dense')
# ax[1].plot(xw_c, cp_c, ':', color='k', label='coarse')
ax[1].plot(xw_l, cp_l, '--', color='k', label='ml')
ax[1].legend(loc='lower right')
ax[1].set_xlabel(f'$x/L$')
ax[1].set_ylabel(f'$C_p$')

ax[2].set_title('Velocity')
cmap = plt.get_cmap("tab10")
for i in range(7):
    label = 'x/c=' + str(xplt[i])
    ax[2].plot(u_d[i,:], z_d[i,:], '-', color=cmap(i), label=label)
    # ax[2].plot(u_c[i,:], z_c[i,:], ':', color=cmap(i))
    ax[2].plot(u_l[i,:], z_l[i,:], '--', color=cmap(i))

ax[2].legend(loc='upper left')
ax[2].set_xlabel(f'$u/U$')
ax[2].set_ylabel(f'$z/L$')

ax[3].set_title('RS')
for i in range(7):
    label = 'x/c=' + str(xplt[i])
    ax[3].plot(r_d[i,:], z_d[i,:], '-', color=cmap(i), label=label)
    # ax[3].plot(r_c[i,:], z_c[i,:], ':', color=cmap(i))
    ax[3].plot(r_l[i,:], z_l[i,:], '--', color=cmap(i))

ax[3].legend(loc='upper left')
ax[3].set_xlabel(f'$R/U^2$')
ax[3].set_ylabel(f'$z/L$')

fig.tight_layout()
plt.savefig(modeldir + '2dwmh.png')
plt.show(block=False)

cf = {
      'x': xw_l,
      'cf': cf_l
     }
cf = pd.DataFrame(cf)
cf.to_csv(modeldir + '2dwmh_cf.dat', sep='\t')

cp = {
      'x': xw_l,
      'cp': cf_l
     }
cp = pd.DataFrame(cp)
cp.to_csv(modeldir + '2dwmh_cp.dat', sep='\t')

for i in range(7):
    vel = {
           'y/c': z_l[i,:],
           'vx, x/c=' + str(xplt[i]) : u_l[i,:]
          }
    vel = pd.DataFrame(vel)
    vel.to_csv(modeldir + '2dwmh_vel'+str(i)+'.dat', sep='\t')

for i in range(7):
    vel = {
           'y/c': z_l[i,:],
           'uv, x/c=' + str(xplt[i]) : u_l[i,:]
          }
    vel = pd.DataFrame(vel)
    vel.to_csv(modeldir + '2dwmh_rs'+str(i)+'.dat', sep='\t')
#
