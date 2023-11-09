import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torchsummary import summary

from unet import UNet_256
from aux_seg import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet_256(n_class=5)  # Initialize an instance of your model
model.load_state_dict(torch.load('./checkpoint/model.pth'))  # Load the model's parameters
model.eval()  # Set the model to evaluation mode if necessary
print(model)


im, masks_array = read_train_all()
images_array = normalize_images(im)

image = np.expand_dims(images_array[25], axis=0)
image = np.expand_dims(image, axis=0)
image = torch.from_numpy(image).float()
print(image.shape)

output = model(image)
print(output.shape)

output_np = output.detach().numpy()

print(output_np.shape)

output_np = np.squeeze(output_np,axis=0)

# print(output_np.shape)

# print(output_np[3].shape)
print("max = ",np.max(output_np[3]))
print("min = ",np.min(output_np[3]))


# treshold = -0.2
# output_np[output_np < treshold ] = 0  



im, msk = read_train_all()
msk = msk.astype(int)
img_nm = normalize_images(im)
msk_sp = spliting_the_segmentation_mask(msk, 5)

a = msk_sp[25,0]
b = msk_sp[25,1]
c = msk_sp[25,2]
d = msk_sp[25,3]
e = msk_sp[25,4]

# plot_images([a,b,c,d,e])

plot_images([output_np[0],a,output_np[1],b,output_np[2],c,output_np[3],d,output_np[3],e,images_array[25], masks_array[25] ])



