import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.model_selection import train_test_split


from aux_seg import *


class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
    
        """
        # # Assuming you have your images and masks in two numpy arrays:
        # # images_array: (250, 256, 256)
        # # masks_array: (250, 256, 256)

        Args:
            images (numpy.ndarray): A numpy array of shape (N, H, W) containing the images.
            masks (numpy.ndarray): A numpy array of shape (N, H, W) containing the segmentation masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Add channel dimension: PyTorch expects images in the shape (C, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        # Convert to PyTorch tensors
        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['mask'] = torch.from_numpy(sample['mask']).long()  # Assuming mask has integer type labels

        return sample['image'], sample['mask']


im, masks_array = read_train_all()
images_array = normalize_images(im)

# # Create the dataset
dataset = CustomDataset(images_array, masks_array)

# spliting the dataset
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)  # 80% for training
val_size = dataset_size - train_size  # Remaining 20% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



# # Create the DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Create data loaders for each split
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)




# test 

print(f"Total dataset size: {dataset_size}")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

print(f"Number of batches in the training data loader: {len(train_loader)}")
print(f"Number of batches in the validation data loader: {len(val_loader)}")

# Fetch a batch from the training data loader
train_images, train_masks = next(iter(train_loader))
print(f"Shape of training images: {train_images.shape}")
print(f"Shape of training masks: {train_masks.shape}")

# Fetch a batch from the validation data loader
val_images, val_masks = next(iter(val_loader))
print(f"Shape of validation images: {val_images.shape}")
print(f"Shape of validation masks: {val_masks.shape}")




