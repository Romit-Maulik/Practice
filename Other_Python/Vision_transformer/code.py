import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, channel_size, hidden_dim, num_layers, num_heads, output_size):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel_size * patch_size ** 2  #
        self.channel_size = channel_size
        
        self.patch_embed = nn.Conv2d(channel_size, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.dropout = nn.Dropout(p=0.1)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        # Convert input images to patches and flatten
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add positional encodings
        x = x + self.positional_encoding
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Compute output
        x = x.mean(dim=1)
        x = self.fc(x)
        
        return x.view(-1, self.channel_size, image_size, image_size)


import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = torch.from_numpy(input_data).float()
        self.output_data = torch.from_numpy(output_data).float()
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, index):
        input_sample = self.input_data[index]
        output_sample = self.output_data[index]
        return input_sample, output_sample

# Example input and output numpy arrays
input_array = np.random.randn(1000, 3, 32, 32)
output_array = np.random.randn(1000, 3, 32, 32)

# Instantiate the custom dataset
dataset = MyDataset(input_array, output_array)

# Create a DataLoader
batch_size = 32
shuffle = True
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Define hyperparameters
image_size = 32
patch_size = 8
hidden_dim = 256
num_layers = 6
num_heads = 8
output_size = image_size ** 2 * 3  # Assumes 3 channel output images
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Initialize model and optimizer
model = VisionTransformer(image_size=image_size, patch_size=patch_size, channel_size=3, hidden_dim=hidden_dim, 
                          num_layers=num_layers, num_heads=num_heads, output_size=output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, data) # Just a dumb identity function
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print("Epoch {:d}, Batch {:d}, Loss: {:.4f}".format(epoch, batch_idx, loss.item()))
