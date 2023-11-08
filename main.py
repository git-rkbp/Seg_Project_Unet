import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader
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


im, np_targets = read_train_all()
np_inputs = normalize_images(im)

images_array = np_inputs
masks_array = np_targets


# # Create the dataset
dataset = CustomDataset(images_array, masks_array)

# # Create the DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Now when you iterate over the dataloader, you will get batches from your array
x, y = next(iter(data_loader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')




