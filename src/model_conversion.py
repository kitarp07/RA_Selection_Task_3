import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

def inflate_conv2d(conv2d, depth):
    conv3d = nn.Conv3d(
    in_channels=conv2d.in_channels,
    out_channels=conv2d.out_channels,
    kernel_size=(depth, *conv2d.kernel_size),
    stride=(1, *conv2d.stride),
    padding=(depth // 2, *conv2d.padding),
    dilation=(1, *conv2d.dilation),
    bias=conv2d.bias is not None
    )
        
    
    with torch.no_grad():
        # Modify the weights without tracking gradients
        weight2d = conv2d.weight.data  # (out, in, h, w)
        weight3d = weight2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth  # (out, in, d, h, w)
        conv3d.weight.copy_(weight3d)
        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias.data)
            
    return conv3d

def inflate_batchnorm2d_to_3d(batchnorm2d):
    return nn.BatchNorm3d(
        num_features=batchnorm2d.num_features,  # Number of channels
        eps=batchnorm2d.eps,                    # Epsilon for numerical stability
        momentum=batchnorm2d.momentum,          # Momentum for running statistics
        affine=batchnorm2d.affine,              # Whether it has learnable parameters
        track_running_stats=batchnorm2d.track_running_stats  # Whether to track running stats
    )
    
def inflate_maxpool2d_to_3d(pool2d_layer):
    return nn.MaxPool3d(kernel_size=(1, pool2d_layer.kernel_size, pool2d_layer.kernel_size),
                        stride=(1, pool2d_layer.stride, pool2d_layer.stride),
                        padding=(0, pool2d_layer.padding, pool2d_layer.padding),
                        dilation=pool2d_layer.dilation, ceil_mode=pool2d_layer.ceil_mode)
    
# Function to inflate AvgPool2D to AvgPool3D
def inflate_avgpool2d_to_3d(pool2d_layer):
    # Create the AvgPool3D layer with the inflated parameters
    return nn.AvgPool3d(kernel_size=(1, pool2d_layer.kernel_size, pool2d_layer.kernel_size), 
                        stride=(1, pool2d_layer.stride, pool2d_layer.stride),
                        padding=(0, pool2d_layer.padding, pool2d_layer.padding))


def inflate_model(model_2d):
    
    inflate_conv = inflate_conv2d(model_2d.features[0], depth=3)
    print(inflate_conv)
    inflate_batch_norm2d = inflate_batchnorm2d_to_3d(model_2d.features[1])
    print(inflate_batch_norm2d)
    inflate_max_pool_2d = inflate_maxpool2d_to_3d(model_2d.features[3])
    print(inflate_max_pool_2d)
    model_2d.features.conv0 = inflate_conv2d(model_2d.features.conv0, depth=3)
    model_2d.features.norm0 = inflate_batchnorm2d_to_3d(model_2d.features.norm0)
    model_2d.features.norm5 = inflate_batchnorm2d_to_3d(model_2d.features.norm5)
    model_2d.features.pool0 = inflate_maxpool2d_to_3d(model_2d.features.pool0)
    
    for block_name in ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']:
        block = getattr(model_2d.features, block_name)
        # print(block)
        for layer_name, layer in block.named_children():
            # print(layer_name, layer)
            if hasattr(layer, 'conv1'):
                # print(layer.conv1)
                layer.conv1 = inflate_conv2d(layer.conv1, depth = 3)
                
            if hasattr(layer, 'conv2'):
                layer.conv2 = inflate_conv2d(layer.conv2, depth=3)
                
            if hasattr(layer, 'norm1'):
                layer.norm1 = inflate_batchnorm2d_to_3d(layer.norm1)
                
            if hasattr(layer, 'norm2'):
                # Inflate BatchNorm2d to BatchNorm3d while keeping learnable parameters
                layer.norm2 = inflate_batchnorm2d_to_3d(layer.norm2)
    
    # Replace convs in transition layers
    for trans_name in ['transition1', 'transition2', 'transition3']:
        trans = getattr(model_2d.features, trans_name)
        if hasattr(trans, 'conv'):
            # print(trans)
            trans.conv = inflate_conv2d(trans.conv, depth=3)
            # print(trans.conv)
            
        if hasattr(trans, 'norm'):
            trans.norm = inflate_batchnorm2d_to_3d(trans.norm)
            # print(trans.norm)
            
        if hasattr(trans, 'pool'):
            # Inflate AvgPool2D to AvgPool3D
            trans.pool = inflate_avgpool2d_to_3d(trans.pool)
            # print(trans.pool)
    
    return model_2d

def extract_feature_map(data, model):

    # Dictionary to store layer names and their corresponding outputs
    layer_outputs = {}

    # Define a hook function to store the feature map along with layer name
    def hook_fn(layer_name):
        def fn(module, input, output):
            layer_outputs[layer_name] = output
        return fn
        
    # Register hooks for the specific layers
    handle_1 = model.features.denseblock4.denselayer16.conv2.register_forward_hook(hook_fn("denseblock4.layer16.conv2"))
    handle_2 = model.features.denseblock4.denselayer15.conv2.register_forward_hook(hook_fn("denseblock4.layer15.conv2"))
    handle_3 = model.features.denseblock4.denselayer14.conv2.register_forward_hook(hook_fn("denseblock4.layer14.conv2"))


    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Turn off gradient calculation (not needed during inference)
        last_conv = model.features(data)  # Pass input through the model
 
    # Output is the feature map or classification result (depending on the model's final layers)
    # print(last_conv.shape)  # Print the output shape to see the result
    
    # Remove hooks after extracting the feature maps
    handle_1.remove()
    handle_2.remove()
    handle_3.remove()
    
    print("Extraction complete")

    return layer_outputs
    
def apply_global_avg_pooling_and_extract_feature_vector(feature_extract):
    gap_output_1 = F.adaptive_avg_pool3d( feature_extract['denseblock4.layer14.conv2'], (1, 1, 1))  
    gap_output_1 = gap_output_1.squeeze()  # Shape will be (32, 1, 1), then remove the 1s
    gap_output_1 = gap_output_1.view(-1)  # Final shape will be (32,)
    
    gap_output_2 = F.adaptive_avg_pool3d( feature_extract['denseblock4.layer15.conv2'], (1, 1, 1))  
    gap_output_2 = gap_output_2.squeeze()  # Shape will be (32, 1, 1), then remove the 1s
    gap_output_2 = gap_output_2.view(-1)  # Final shape will be (32,)

    gap_output_3 = F.adaptive_avg_pool3d( feature_extract['denseblock4.layer16.conv2'], (1, 1, 1))  
    gap_output_3 = gap_output_3.squeeze()  # Shape will be (32, 1, 1), then remove the 1s
    gap_output_3 = gap_output_3.view(-1)  # Final shape will be (32,)
    
    return gap_output_1, gap_output_2, gap_output_3


    