import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.ndimage as ndimage
from skimage.morphology import ball, remove_small_objects, binary_dilation, binary_erosion, disk
import torch

def preprocess_mask(bone_mask):
    """
    Preprocess a binary bone mask to remove noise, fill holes, 
    and separate femur and tibia by removing the connecting slice.

    Steps:
    1. Remove small connected components (noise, fibula, etc.)
    2. Fill internal holes in axial slices
    3. Disconnect femur and tibia by zeroing out a thin axial region

    Parameters:
        bone_mask (np.ndarray): 3D binary mask (e.g., from HU thresholding)

    Returns:
        np.ndarray: Refined binary mask with femur and tibia separated
    """
    # Step 1: Remove small objects to keep only major bone structures

    bone_outer = remove_small_objects(bone_mask, min_size=500)
    
    # Step 2: Fill holes slice-by-slice (axial)
    filled_mask = np.zeros_like(bone_outer)
    for z in range(bone_outer.shape[2]):  # axial slices
        filled_mask[:, :, z] = ndimage.binary_fill_holes(bone_outer[:, :, z])
    
    # Step 3: Cut axial slice(s) to disconnect femur and tibia

    cut_mask = filled_mask.copy() # Copy the filled mask
    cut_mask[:,:,105:106] = 0  # Zero out that connecting region to separate out femur and tibia
    
    return cut_mask

def threshold_ct(vol_data_array):
    """
    Apply intensity thresholding to extract bone structures.

    Parameters:
        vol_data_array (np.ndarray): 3D CT volume in HU
        threshold (int): HU threshold for bone (default: 320)

    Returns:
        np.ndarray: Binary mask (True where HU > threshold)
    """
    bone_mask = (vol_data_array > 320)
    return bone_mask 

def fill_hole_component_1(component_1):
    """
    Fill internal holes in a 3D binary mask along selected planes.

    Parameters:
        mask (np.ndarray): 3D binary mask (dtype=bool or 0/1)
        
    Returns:
        np.ndarray: Binary mask with filled internal holes.
    """
    
    mask = component_1.copy()

    for x in range(component_1.shape[0]):  # loop through sagittal slices
        mask[x, :, :] = ndimage.binary_fill_holes(mask[x, :, :])

    for z in range(component_1.shape[2]):  # loop through axial slices
        mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])

    for y in range(component_1.shape[1]):  # loop through axial slices
        mask[:, y, :] = ndimage.binary_fill_holes(mask[:, y, :])
    
    return mask

def connect_edges_component_2(component_2):
    """
    Fill internal holes in a 3D binary mask along selected planes.

    Parameters:
        mask (np.ndarray): 3D binary mask (dtype=bool or 0/1)
        
    Returns:
        np.ndarray: Binary mask with filled internal holes.
    """
    
    mask = component_2.copy()

    for z in range(mask.shape[2]):
        slice_2d = mask[:, :, z] 

        # Apply mild dilation to connect broken rim
        dilated = ndimage.binary_dilation(slice_2d, structure=disk(1), iterations=3)
        eroded = ndimage.binary_erosion(dilated, structure=disk(1), iterations=1)
        # Fill enclosed holes
        filled = ndimage.binary_fill_holes(dilated)

        mask[:, :, z] = filled
            
    return mask

def fill_and_smooth(component_2):
    """
    Fill holes and smooth the edges of a 3D binary mask using 2D filling and 3D morphology.

    Steps:
    1. Fill holes along coronal and axial slices
    2. Apply binary dilation followed by erosion (morphological closing) to smooth edges

    Parameters:
        component_mask (np.ndarray): 3D binary mask (dtype=bool or 0/1)

    Returns:
        np.ndarray: Processed binary mask with filled holes and smoothed edges
    """
    
    mask = component_2.copy()

    for y in range(component_2.shape[1]):  
        mask[:, y, :] = ndimage.binary_fill_holes(mask[:, y, :])
        
    for z in range(component_2.shape[2]):  
        mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
        
    # smoothing out the edges
    mask = binary_dilation(mask, footprint=ball(1))
    mask = binary_erosion(mask, footprint=ball(1))
    
    return mask
            

def get_largest_components(bone_mask):
    """
    Identify and extract the two largest connected components from a 3D binary mask.

    This is used to separate the femur and tibia after masking bones 
    from a CT scan using thresholding and preprocessing for further processing on separated components.

    Parameters:
        bone_mask (np.ndarray): 3D binary mask (dtype=bool or 0/1)

    Returns:
        tuple: Two binary masks (np.ndarray) of the two largest components
    """
    # label connected components in the mask
    labeled_mask, num_components = ndimage.label(bone_mask)
    
    # Compute size (voxel_count_ of each component (starting from label 1)
    component_sizes = ndimage.sum(np.ones_like(labeled_mask), labeled_mask, index=np.arange(1, num_components + 1))

    # Sort labels by size in descending order; returns indicies
    sorted_indices = np.argsort(component_sizes)[::-1]  # largest first

    # Get the labels of the two largest components
    largest_label_1 = sorted_indices[0] + 1  # +1 because labels start at 1
    largest_label_2 = sorted_indices[1] + 1
    
    #extract binary masks for each component
    largest_component_1 = (labeled_mask == largest_label_1)
    largest_component_2 = (labeled_mask == largest_label_2)
    
    return largest_component_1, largest_component_2

def combine_mask(component_1, component_2):
    combined_mask = component_1 | component_2
    return combined_mask

def get_labelled_mask(combined_mask):
    
    labeled_mask, num_components = ndimage.label(combined_mask)


    # Step 2: Compute size of each component (excluding background)
    component_sizes = ndimage.sum(np.ones_like(labeled_mask), labeled_mask, index=np.arange(1, num_components + 1))

    # Step 3: Get labels of two largest components
    sorted_indices = np.argsort(component_sizes)[::-1]
    largest_label_1 = sorted_indices[0] + 1  # +1 because labels start at 1
    largest_label_2 = sorted_indices[1] + 1

    # Step 4: Use center of mass to determine which is tibia (lower Z)
    com1 = ndimage.center_of_mass(labeled_mask == largest_label_1)
    com2 = ndimage.center_of_mass(labeled_mask == largest_label_2)

    if com1[2] < com2[2]:
        tibia_label = largest_label_1
        femur_label = largest_label_2
    else:
        tibia_label = largest_label_2
        femur_label = largest_label_1

    # Step 5: Create labeled mask: 1 = femur, 2 = tibia
    final_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
    final_mask[labeled_mask == tibia_label] = 2
    final_mask[labeled_mask == femur_label] = 1
    
    return final_mask

def crop_vol(img_array):
    return img_array[256:512, 200:456, :]
    
def extract_roi(ct_img_array, mask_img_array, label):
    roi = np.where(mask_img_array == label, ct_img_array, 0)
    return roi

def normalize_volume(ct_img_array, roi_img_array):
    """
    Normalizes the volume data (e.g., tibia, femur, background) based on the min and max values of the CT image.
    
    Parameters:
    - ct_img_array: The CT scan image array, used for finding the min and max values.
    - volume_data: The volume data (e.g., tibia, femur, background) to be normalized.
    
    Returns:
    - The normalized volume data.
    """

    normalized_img_array = (roi_img_array - ct_img_array.min()) / (ct_img_array.max() - ct_img_array.min())
    normalized_img_array = torch.tensor(normalized_img_array)
    return normalized_img_array


def prepare_roi_for_model(roi_img_array):
    """
    Prepares a volume data for model input by reshaping and replicating channels as needed.
    
    Parameters:
    - volume_data: The volume data (e.g., tibia_volume) to be prepared. 
      This should be a 3D tensor (e.g., [216, 512, 512]).
    
    Returns:
    - The volume data reshaped and replicated to fit the model's expected input format.
      (Shape: [1, 3, 216, 512, 512])
    """
        # Permute to change the shape to (216, 512, 512)
    volume_data = roi_img_array.permute(2, 0, 1)  # Shape will now be (216, 512, 512)
    
    # Replicate the single channel along the 3 channel axis
    volume_data = volume_data.unsqueeze(0).repeat(3, 1, 1, 1)  # Shape will now be (3, 216, 512, 512)
    
    # Add batch dimension
    volume_data = volume_data.unsqueeze(0)  # Shape will now be (1, 3, 216, 512, 512)
    
    # Convert to float (important for model input)
    volume_data = volume_data.float()
    
    return volume_data


    


    