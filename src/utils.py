import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
def load_nifti(path_to_data):
    """
    Load a NIfTI file and return the volume data and affine.
    
    Parameters:
        path (str): Path to the .nii or .nii.gz file

    Returns:
        tuple: (3D numpy array, affine matrix)
    """
    data = nib.load(path_to_data)
    
    return data

def get_img_array(nifti_img):
    """
    Extract the image data array from a NIfTI image.

    Parameters:
        nifti_img (nib.Nifti1Image): Loaded NIfTI image

    Returns:
        np.ndarray: 3D volume data as a NumPy array
    """
    vol_data = np.array(nifti_img.get_fdata())
    return vol_data

def plot_slices(data, x=375, y=256, z=108):
    """
    Plot axial, sagittal, and coronal slices of a 3D volume.

    Parameters:
        data (np.ndarray): 3D volume data (e.g., CT or mask)

    Notes:
        - Slice indices are hardcoded for demonstration.
        - Axial:     data[:, :, z]
        - Sagittal:  data[x, :, :]
        - Coronal:   data[:, y, :]
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice 
    axes[0].imshow(data[:, :, z], cmap='gray')
    axes[0].set_title('Axial Slice (Index {z})')
    axes[0].axis('off')

    # Coronal slice
    axes[1].imshow(data[x, :, :], cmap='gray')
    axes[1].set_title('Saggital Slice (Index {x})')
    axes[1].axis('off')
    
    # Sagittal slice
    axes[2].imshow(data[:, y, :], cmap='gray')
    axes[2].set_title('Coronal Slice (Index {y})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# Function to compute cosine similarity
def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.view(1, -1), v2.view(1, -1))

def plot_cropped_slices(data, z=80, x=60, y=115):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice (z-plane)
    axes[0].imshow(data[:, :, z], cmap='gray')
    axes[0].set_title(f'Axial Slice (Index {z})')
    axes[0].axis('off')

    
    # Sagittal slice (y-plane)
    axes[2].imshow(data[y, :, :], cmap='gray')
    axes[2].set_title(f'Sagittal Slice (Index {y})')
    axes[2].axis('off')

    # Coronal slice (x-plane)
    axes[1].imshow(data[:, x, :], cmap='gray')
    axes[1].set_title(f'Coronal Slice (Index {x})')
    axes[1].axis('off')


    plt.tight_layout()
    plt.show()



def save_similarity_values_to_csv_file(*args):
    data = {
    # Tibia (last) vs Femur (last, third_last, fifth_last)
    "Tibia(last)-Femur(last,3rd, 5th)": [
            args[0],  # similarity_tibia_last_femur_last
            args[1],  # similarity_tibia_last_femur_3rd_last
            args[2]   # similarity_tibia_last_femur_5th_last
    ],
    "Tibia(3rd_last)-Femur(last,3rd, 5th)": [
            args[3],  # similarity_tibia_3rd_last_femur_last
            args[4],  # similarity_tibia_3rd_last_femur_3rd_last
            args[5]   # similarity_tibia_3rd_last_femur_5th_last
    ],
    "Tibia(5th_last)-Femur(last,3rd, 5th)": [
            args[6],  # similarity_tibia_5th_last_femur_last
            args[7],  # similarity_tibia_5th_last_femur_3rd_last
            args[8]   # similarity_tibia_5th_last_femur_5th_last
    ],

    # Tibia (last) vs Background (last, third_last, fifth_last)
    "Tibia(last)-Background(last,3rd, 5th)": [
            args[9],  # similarity_tibia_last_background_last
            args[10], # similarity_tibia_last_background_3rd_last
            args[11]  # similarity_tibia_last_background_5th_last
    ],
    "Tibia(3rd_last)-Background(last,3rd, 5th)": [
            args[12], # similarity_tibia_3rd_last_background_last
            args[13], # similarity_tibia_3rd_last_background_3rd_last
            args[14]  # similarity_tibia_3rd_last_background_5th_last
    ],
    "Tibia(5th_last)-Background(last,3rd, 5th)": [
            args[15], # similarity_tibia_5th_last_background_last
            args[16], # similarity_tibia_5th_last_background_3rd_last
            args[17]  # similarity_tibia_5th_last_background_5th_last
    ],

    # Femur (last) vs Background (last, third_last, fifth_last)
    "Femur(last)-Background(last,3rd, 5th)": [
            args[18], # similarity_femur_last_background_last
            args[19], # similarity_femur_last_background_3rd_last
            args[20]  # similarity_femur_last_background_5th_last
    ],
    "Femur(3rd_last)-Background(last,3rd, 5th)": [
            args[21], # similarity_femur_3rd_last_background_last
            args[22], # similarity_femur_3rd_last_background_3rd_last
            args[23]  # similarity_femur_3rd_last_background_5th_last
    ],
    "Femur(5th_last)-Background(last,3rd, 5th)": [
            args[24], # similarity_femur_5th_last_background_last
            args[25], # similarity_femur_5th_last_background_3rd_last
            args[26]  # similarity_femur_5th_last_background_5th_last
    ],
    }
    
    similarity_df = pd.DataFrame(data)
        # Save to CSV
    similarity_df.to_csv('C:/Users/DELL/Desktop/RA_Selection_Task_3/results/similarity.csv', index=False)
    print("Similarities saved to similarity.csv")


    
    
def save_file(final_mask, original_image_vol, file_name):
    labelled_mask = final_mask.astype(np.uint8)  # NIfTI format expects int types
    nii_mask = nib.Nifti1Image(labelled_mask, affine=original_image_vol.affine)
    nib.save(nii_mask, Path("results")/file_name)


