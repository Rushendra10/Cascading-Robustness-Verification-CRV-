import numpy as np
import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# def mnist_loaders(batch_size): 
#     mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
#     mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
#     train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=False)
#     test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=False)
#     return train_loader, test_loader


def custom_mnist_loaders(folder_path, batch_size):
    transform = transforms.ToTensor()  # Keep transformations the same

    # Load all .npy files
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])
    
    print(f"Num files is {len(files)}")

    data_list, label_list, filename_list = [], [], []
    for i, file in enumerate(files):
        # print(f"Input number {i} with filename {file}")
        data = np.load(file)  # Load the .npy file
        if data.ndim == 2:  # Ensure correct shape
            data = np.expand_dims(data, axis=0)  # Add channel dimension

        data = transform(data)  # Apply transforms
        data_list.append(data)
        filename_list.append(file)
        # Extract label from filename (assuming format "image_index_label.npy")
        label = int(file.split('_')[-1].split('.')[0])  # Extracts label
        label_list.append(label)

    # Stack inputs and convert labels to tensors
    dataset = TensorDataset(torch.stack(data_list), torch.tensor(label_list, dtype=torch.long))

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    
    print(f"Len loader is {len(loader)}")
    

    shuffled_filenames = []
    for batch_indices in loader.batch_sampler:
        shuffled_filenames.extend([filename_list[idx] for idx in batch_indices])

    return loader, shuffled_filenames


def mnist_model(): 
    model = nn.Sequential(
    nn.Flatten(),                  # Convert (1, 28, 28) to (784)
    nn.Linear(28*28, 500),         # Fully connected layer with 500 neurons
    nn.ReLU(),                     # Activation function
    nn.Linear(500, 10)             # Output layer (10 classes for MNIST)
)
    return model
