import numpy as np
import torch
from graph_weather import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import os, sys, argparse

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        # Load the data from the file
        self.data = torch.from_numpy(np.load(filename).reshape(-1,256*128,1).astype('float32'))
        print('Loaded data from filename:',filename)
        sys.stdout.flush()

    def __getitem__(self, index):
        # Get the input and output values from the data
        input_value = self.data[index]
        output_value = self.data[index+1]

        # Return the input and output as tensors
        return input_value, output_value

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)-1 # This to make sure you don't access the last value of the index
    
def train(gpu, args):
    
    rank = args.node_id *  args.gpus + gpu
    
    print('Hello from rank:',rank)
    sys.stdout.flush()
    
    # Set up distributed training environment
    dist.init_process_group(backend='nccl', init_method="env://", world_size=args.world_size, rank=rank)
    
    print('Initialized distributed training environment')
    sys.stdout.flush()

    data_dir = '/lcrc/project/MultiscaleGNN/data_swing/'
    # Set up the dataset and dataloader
    dataset = CustomDataset(data_dir+'snapshot_'+f'{rank:04d}' + '_swing.npy')
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
    
    lat_lons = []
    lats = np.load(data_dir+'lat.npy')
    lons = np.load(data_dir+'lon.npy')

    for lat in range(np.shape(lats)[0]):
        for lon in range(np.shape(lons)[0]):
            lat_lons.append((lats[lat], lons[lon]))
    
    print('Initializing models on GPU')
    sys.stdout.flush()
    
    # Set up model and send to GPU
    model = GraphWeatherForecaster(lat_lons,feature_dim=1,aux_dim=0)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    print('Now using device:',torch.cuda.current_device(),', with rank:',rank)
    sys.stdout.flush()
    
    # Set up the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((1,)))
    
    # Wrap model
    model = DDP(model, device_ids=[gpu])
    print('Model now wrapped with DDP')
    sys.stdout.flush()
    
    # Training loop
    for epoch in range(args.epochs):
        print('Training epoch number:',epoch)
        sys.stdout.flush()
        for inputs, targets in dataloader:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0 and epoch % 2 == 0:
                print('Batch loss:',loss.item(), ', for epoch:',epoch)
                sys.stdout.flush()
            sys.stdout.flush()
        
        dist.barrier()
        
        if rank == 0:
            model_dir = '/lcrc/project/MultiscaleGNN/'
            dict_model = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(dict_model, model_dir+'/train_model.pth')
            
        dist.barrier()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=2, type=int, metavar='N',
                        help='number of nodes (default: 2)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--ip_address', type=str, required=True,
                        help='ip address of the host node')
    
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.ip_address
    os.environ['MASTER_PORT'] = '8888'
    args.node_id = int(os.environ.get("SLURM_NODEID"))
    
    print('Running trainings with:')
    print('Nodes:',args.nodes)
    print('GPUs per node:',args.gpus)
    print('Epochs:',args.epochs)
    print('World size:',args.world_size)
    print('IP Address:',args.ip_address)
    print('Slurm node ID:',args.node_id)
    
    mp.spawn(train, args=(args,), nprocs=args.gpus)
    
if __name__ == '__main__':
    main()


