from src.utils import load_nifti, save_similarity_values_to_csv_file, plot_slices, get_img_array, cosine_similarity, save_file, plot_cropped_slices
from src.model_conversion import inflate_model, extract_feature_map, apply_global_avg_pooling_and_extract_feature_vector
from src.preprocessing import (preprocess_mask, threshold_ct, fill_hole_component_1, connect_edges_component_2, 
                              fill_and_smooth, get_largest_components, get_labelled_mask, combine_mask,
                              extract_roi, crop_vol, prepare_roi_for_model, normalize_volume)

import torchvision.models as models

def main():
    path_to_data = "./data/3702_left_knee.nii.gz"
    
    #load data
    ct_vol = load_nifti(path_to_data)
    ct_images_vol_data = get_img_array(ct_vol)
            
    #thresholding to get bone mask
    bone_mask = threshold_ct(ct_images_vol_data)
        
    #preprocess bone mask to fill holes, remove noise and disconnect femur and tibia
    bone_mask_preprocessed = preprocess_mask(bone_mask)
    
    # extract two largest components from mask and separate them for further preprocessing
    component_1, component_2 = get_largest_components(bone_mask_preprocessed)

    # fill internal holes in the femur
    component_1_filled = fill_hole_component_1(component_1)
    
    #connect edges in broken rim of tibia
    component_2_filled = connect_edges_component_2(component_2)

    
    # fill holes and smooth the rough edges
    smoothed_component_2 = fill_and_smooth(component_2_filled)

    # combine the preprocessed separate components into 1 mask
    combined_mask = combine_mask(component_1_filled, smoothed_component_2)

    # label the components and get final mask
    final_mask = get_labelled_mask(combined_mask)
    # plot_slices(final_mask)
    #save mask to results folder
    save_file(final_mask, ct_vol, "segmentation_mask.nii.gz")
    
    #split tibia
    tibia_volume = extract_roi(ct_images_vol_data, final_mask, label=2)
    tibia_volume = crop_vol(tibia_volume)
    # plot_cropped_slices(tibia_volume)

    #split femur
    femur_volume = extract_roi(ct_images_vol_data, final_mask, label=1)
    femur_volume = crop_vol(femur_volume)
    # plot_cropped_slices(femur_volume)
    
    background_volume = extract_roi(ct_images_vol_data, final_mask, label=0)
    background_volume = crop_vol(background_volume)
    # plot_cropped_slices(background_volume)
    
    #normalize cropped data
    
    
    tibia_volume = normalize_volume(ct_images_vol_data, tibia_volume)
    femur_volume = normalize_volume(ct_images_vol_data, femur_volume)
    background_volume = normalize_volume(ct_images_vol_data, background_volume)

    #prepare roi for model
    tibia_volume = prepare_roi_for_model(tibia_volume)
    femur_volume = prepare_roi_for_model(femur_volume)
    background_volume = prepare_roi_for_model(background_volume)
    
    print(tibia_volume.shape)


    
    #Segmentation Splitting Complete
    
    # Task 2
    
    model_2d = models.densenet121(pretrained=True)
    #inflate model
    model_3d = inflate_model(model_2d)
    print(model_3d)
    
    # Task 3 feature extraction
    
    # extract last, 3rd last, 5th last features for tibia
    tibia_extract = extract_feature_map(tibia_volume, model_3d)
    
    # extract last, 3rd last, 5th last features for femur
    femur_extract = extract_feature_map(femur_volume, model_3d)
    
    # extract last, 3rd last, 5th last features for background
    background_extract = extract_feature_map(background_volume, model_3d)
    
    # apply average global pooling and extract feature vector
    tibia_extract_fifth_last, tibia_extract_third_last, tibia_extract_last = apply_global_avg_pooling_and_extract_feature_vector(tibia_extract)
    # print(tibia_extract_fifth_last.shape, tibia_extract_third_last.shape, tibia_extract_last.shape)
    
    femur_extract_fifth_last, femur_extract_third_last, femur_extract_last = apply_global_avg_pooling_and_extract_feature_vector(femur_extract)
    # print(femur_extract_fifth_last.shape, femur_extract_third_last.shape, femur_extract_last.shape)
    
    background_extract_fifth_last, background_extract_third_last, background_extract_last = apply_global_avg_pooling_and_extract_feature_vector(background_extract)
    # print(background_extract_fifth_last.shape, background_extract_third_last.shape, background_extract_last.shape)
    
    # TASK 4 Feature Comparison
    similarity_tibia_last_femur_last = cosine_similarity(tibia_extract_last, femur_extract_last)
    similarity_tibia_last_femur_3rd_last = cosine_similarity(tibia_extract_last, femur_extract_third_last)
    similarity_tibia_last_femur_5th_last = cosine_similarity(tibia_extract_last, femur_extract_fifth_last)

    similarity_tibia_3rd_last_femur_last = cosine_similarity(tibia_extract_third_last,femur_extract_last)
    similarity_tibia_3rd_last_femur_3rd_last = cosine_similarity(tibia_extract_third_last, femur_extract_third_last)
    similarity_tibia_3rd_last_femur_5th_last = cosine_similarity(tibia_extract_third_last, femur_extract_fifth_last)

    similarity_tibia_5th_last_femur_last = cosine_similarity(tibia_extract_fifth_last, femur_extract_last)
    similarity_tibia_5th_last_femur_3rd_last = cosine_similarity(tibia_extract_fifth_last, femur_extract_third_last)
    similarity_tibia_5th_last_femur_5th_last = cosine_similarity(tibia_extract_fifth_last, femur_extract_fifth_last)


    similarity_tibia_last_background_last = cosine_similarity(tibia_extract_last, background_extract_last)
    similarity_tibia_last_background_3rd_last = cosine_similarity(tibia_extract_last, background_extract_third_last)
    similarity_tibia_last_background_5th_last = cosine_similarity(tibia_extract_last, background_extract_fifth_last)

    similarity_tibia_3rd_last_background_last = cosine_similarity(tibia_extract_third_last, background_extract_last)
    similarity_tibia_3rd_last_background_3rd_last = cosine_similarity(tibia_extract_third_last, background_extract_third_last)
    similarity_tibia_3rd_last_background_5th_last = cosine_similarity(tibia_extract_third_last, background_extract_fifth_last)

    similarity_tibia_5th_last_background_last = cosine_similarity(tibia_extract_fifth_last, background_extract_last)
    similarity_tibia_5th_last_background_3rd_last =  cosine_similarity(tibia_extract_fifth_last, background_extract_third_last)
    similarity_tibia_5th_last_background_5th_last =  cosine_similarity(tibia_extract_fifth_last, background_extract_fifth_last)


    similarity_femur_last_background_last = cosine_similarity(femur_extract_last, background_extract_last)
    similarity_femur_last_background_3rd_last = cosine_similarity(femur_extract_last, background_extract_third_last)
    similarity_femur_last_background_5th_last = cosine_similarity(femur_extract_last, background_extract_fifth_last)

    similarity_femur_3rd_last_background_last = cosine_similarity(femur_extract_third_last, background_extract_last)
    similarity_femur_3rd_last_background_3rd_last = cosine_similarity(femur_extract_third_last, background_extract_third_last)
    similarity_femur_3rd_last_background_5th_last = cosine_similarity(femur_extract_third_last, background_extract_fifth_last)

    similarity_femur_5th_last_background_last = cosine_similarity(femur_extract_fifth_last, background_extract_last)
    similarity_femur_5th_last_background_3rd_last = cosine_similarity(femur_extract_fifth_last, background_extract_third_last)
    similarity_femur_5th_last_background_5th_last = cosine_similarity(femur_extract_fifth_last, background_extract_fifth_last)
    
    save_similarity_values_to_csv_file(similarity_tibia_last_femur_last, similarity_tibia_last_femur_3rd_last, similarity_tibia_last_femur_5th_last,
                                       similarity_tibia_3rd_last_femur_last, similarity_tibia_3rd_last_femur_3rd_last, similarity_tibia_3rd_last_femur_5th_last,
                                       similarity_tibia_5th_last_femur_last, similarity_tibia_5th_last_femur_3rd_last, similarity_tibia_5th_last_femur_5th_last,
                                       similarity_tibia_last_background_last, similarity_tibia_last_background_3rd_last, similarity_tibia_last_background_5th_last,
                                       similarity_tibia_3rd_last_background_last, similarity_tibia_3rd_last_background_3rd_last, similarity_tibia_3rd_last_background_5th_last,
                                       similarity_tibia_5th_last_background_last, similarity_tibia_5th_last_background_3rd_last, similarity_tibia_5th_last_background_5th_last,
                                       similarity_femur_last_background_last, similarity_femur_last_background_3rd_last, similarity_femur_last_background_5th_last,
                                       similarity_femur_3rd_last_background_last, similarity_femur_3rd_last_background_3rd_last, similarity_femur_3rd_last_background_5th_last,
                                       similarity_femur_5th_last_background_last, similarity_femur_5th_last_background_3rd_last, similarity_femur_5th_last_background_5th_last
                                       )
    
    # TASK five complete