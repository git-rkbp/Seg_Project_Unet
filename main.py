import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils import data


from aux_seg import *



class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs,  # NumPy array of shape (250, 256, 256)
                 targets, # NumPy array of shape (250, 256, 256)
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        x = self.inputs[index]
        y = self.targets[index]

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # If the array has only one channel and you want to add a channel dimension:
        # x = x[None, ...]  # This adds a channel dimension (assuming channel-first convention)
        # y = y[None, ...]  # This adds a channel dimension (assuming channel-first convention)

        # Typecasting
        x = torch.from_numpy(x).type(self.inputs_dtype)
        y = torch.from_numpy(y).type(self.targets_dtype)

        return x, y

# Assume np_inputs is your input images array of shape (250, 256, 256)
# and np_targets is your target images array of the same shape
# np_inputs = ...
# np_targets = ...



im, np_targets = read_train_all()
np_inputs = normalize_images(im)





training_dataset = SegmentationDataSet(inputs=np_inputs,
                                       targets=np_targets,
                                       transform=None)

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=2,
                                      shuffle=True)

# Now when you iterate over the dataloader, you will get batches from your array
x, y = next(iter(training_dataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')
# x = shape: torch.Size([2, 256, 256]); type: torch.float32
# y = shape: torch.Size([2, 256, 256]); class: tensor([0, 1, 2, 3, 4]); type: torch.int64


