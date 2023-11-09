import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.model_selection import train_test_split
import torch.optim as optim



from unet import UNet_256
from aux_seg import *


class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = np.expand_dims(image, axis=0)

        # Ensure that the mask has the correct shape [batch_size, height, width]
        if mask.shape[0] == 1:
            mask = mask[0]


        mask = mask.astype(np.float32)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['mask'] = torch.from_numpy(sample['mask']).long()

        return sample['image'], sample['mask']



NUM_OF_CLASSES = 5

images_array, masks_array = read_train_all()
images_array = normalize_images(images_array)
# masks_array = spliting_the_segmentation_mask(masks_array, NUM_OF_CLASSES)

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Create an instance of your UNet model
model = UNet_256(n_class=5).to(device)

# Define the loss function (e.g., Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()


# Define the optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the LR scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # Set the model in training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)

    # LR scheduler step
    scheduler.step()
    
    # Validation part
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():  # No need to track the gradients
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}")

print("Training finished")
torch.save(model.state_dict(), './checkpoint/model.pth')
