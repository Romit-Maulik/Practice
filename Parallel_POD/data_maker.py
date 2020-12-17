import numpy as np

# Data too large to put on repo
# Contact me (Romit Maulik) if you want to take a look at it. In principle, easy to use your own data set.

# data = np.load('snapshot_matrix_pod.npy')[:4096*3,:900] # num_dofs x num_snapshots
# data_mean = np.mean(data,axis=1).reshape(-1,1)
# data = data - data_mean

# # Compute POD using method of snapshots (only once for comparison)
# from Dual_POD import generate_pod_bases
# serial_bases = generate_pod_bases(data)
# np.save('Serial_POD_Bases.npy',serial_bases)

# ppr = int(data.shape[0]/6)
# for rank in range(6):
#     temp = data[ppr*rank:ppr*(rank+1)]
#     np.save('points_rank_'+str(rank)+'.npy',temp)


# data = np.load('snapshot_matrix_pod.npy')[:4096*3,:900] # num_dofs x num_snapshots
# data_mean = np.mean(data,axis=1).reshape(-1,1)
# data = data - data_mean

# spr = int(data.shape[1]/6)
# for rank in range(6):
#     temp = data[:,spr*rank:spr*(rank+1)]
#     np.save('snapshots_rank_'+str(rank)+'.npy',temp)

# Check POD basis functions
import matplotlib.pyplot as plt

mode_num = 2

data = np.load('Dual_POD_Basis.npy')
plt.figure()
plt.imshow(data[:4096,mode_num].reshape(64,64))
plt.colorbar()
plt.savefig('Dual_POD_Mode2.png')

data = np.load('APMOS_Basis.npy')
plt.figure()
plt.imshow(data[:4096,mode_num].reshape(64,64))
plt.colorbar()
plt.savefig('APMOS_Mode2.png')

data = np.load('Serial_POD_Bases.npy')
plt.figure()
plt.imshow(data[:4096,mode_num].reshape(64,64))
plt.colorbar()
plt.savefig('Serial_Mode2.png')
plt.show()

