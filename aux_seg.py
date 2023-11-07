import nibabel
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize 



train_ids = [1,2,3,5,8,10,13,19]
test_ids = [21,22,32,39]

def read_train_exam(exam_nb):
    image = nibabel.load('./chaos/train/%02d-T2SPIR-src.nii.gz'%(exam_nb))
    mask = nibabel.load('./chaos/train/%02d-T2SPIR-mask.nii.gz'%(exam_nb))
    return np.copy(image.get_fdata()), np.copy(mask.get_fdata())

def read_test_exam(exam_nb):
    image = nibabel.load('./chaos/test/%02d-T2SPIR-src.nii.gz'%(exam_nb))
    return np.copy(image.get_fdata())
    
def read_train_all():
    # read the 8 train images and return as a np structure
    imgs, masks = [], []
    for train_id in train_ids:
        image, mask = read_train_exam(train_id)
        num_images = image.shape[2] 
       
        resized_images = np.zeros((256, 256, num_images))
        resized_masks = np.zeros((256, 256, num_images))
        for i in range(num_images):
            
            resized_images[:,:,i] = resize(image[:,:,i],(256,256),mode='reflect',anti_aliasing=True)
            resized_masks[:,:,i] = resize(mask[:,:,i],(256,256),mode='reflect',anti_aliasing=True)

        imgs.append(np.copy(resized_images))
        masks.append(np.copy(resized_masks))

    imgs_concatenated = np.concatenate(imgs, axis=2)
    mask_concatenated = np.concatenate(masks, axis=2)

    im_tp = np.transpose(imgs_concatenated, (2, 0, 1))
    ms_tp = np.transpose(mask_concatenated, (2, 0, 1))

    return im_tp, ms_tp  

def plot_images(images_list):
    
    """
    Plots all the images in the given list.
    
    Args:
    - images_list (list): A list containing individual image arrays.
    """
    
    num_images = len(images_list)
    
    # Calculate number of rows and columns for the grid of images
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    fig, axarr = plt.subplots(rows, cols, figsize=(8, 8))
    
    # Ensure axarr is always a 2D array
    if rows == 1 and cols == 1:
        axarr = np.array([[axarr]])
    elif rows == 1 or cols == 1:
        axarr = axarr.reshape(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                axarr[i, j].imshow(images_list[idx], cmap='gray')
                axarr[i, j].axis('off')
            else:
                axarr[i, j].axis('off')  # Hide axes if there are no more images
                
    
    plt.tight_layout()
    plt.show()


def normalize_images(images):
    num_images = images.shape[0]
    normalzed_array = np.zeros(images.shape)

    for i in range(num_images):
        normalzed_array[i] = np.interp(images[i], (images[i].min(), images[i].max()), (0, 1))
        
    return normalzed_array



im, ms = read_train_all()
nm_im = normalize_images(im)

print(nm_im.min(), nm_im.max())

print("hello")


print(im)






