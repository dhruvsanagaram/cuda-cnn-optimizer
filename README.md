# CUDA Convolution Forward Pass

This project implements a GPU-accelerated forward pass for a convolutional neural network using CUDA. The code is designed to efficiently compute the forward convolution over a batch of input images using a shared kernel (mask) and generate the output feature maps.

## Table of Contents
1. [Introduction](#introduction)
2. [Code Structure](#code-structure)
3. [Kernel Implementation](#kernel-implementation)
4. [Usage](#usage)
5. [CUDA Details](#cuda-details)
6. [Error Handling](#error-handling)
7. [Performance Considerations](#performance-considerations)

## Introduction

In this project, we use CUDA to implement the forward pass of a convolutional neural network (CNN) over a batch of images. The convolution operation is a crucial part of many deep learning models and is typically computationally intensive, making it a great candidate for GPU acceleration.

### Problem Overview

The convolution operation is performed on a batch of input images with a set of kernels (masks) to produce a set of output feature maps. Each image in the batch is processed with multiple kernels, and the result is stored in the output array. The implementation supports configurable batch sizes, input feature map channels, output feature maps, kernel sizes, and strides.

### Input Parameters:
- **Input (B, C, H, W)**: A batch of input images, where:
  - `B`: Batch size
  - `C`: Number of input feature maps (channels)
  - `H`: Height of the input image
  - `W`: Width of the input image
- **Mask (M, C, K, K)**: Convolution kernels (filters), where:
  - `M`: Number of output feature maps
  - `C`: Number of input feature maps (channels)
  - `K`: Height and width of the square kernel
- **Stride (S)**: Stride length for the convolution operation
- **Output (B, M, H_out, W_out)**: Resulting feature maps after convolution, where:
  - `H_out = (H - K) / S + 1`: Output height
  - `W_out = (W - K) / S + 1`: Output width

## Code Structure

The main components of this project include:

1. **`conv_forward_kernel`**: The CUDA kernel that performs the forward convolution operation on the GPU.
2. **`conv_forward_gpu_prolog`**: Sets up memory allocations and copies input data from the host to the device.
3. **`conv_forward_gpu`**: Launches the convolution kernel and synchronizes the GPU.
4. **`conv_forward_gpu_epilog`**: Copies the result back to the host and frees device memory.
5. **`get_device_properties`**: Prints out the properties of the available CUDA devices.

## Kernel Implementation

### `conv_forward_kernel`

This function is where the main convolution operation takes place. It computes the convolution for each image in the batch across all output feature maps by iterating over the input channels, applying the kernel, and summing up the results.

- **Indexing Macros**: The `out_4d`, `in_4d`, and `mask_4d` macros simplify accessing 4D arrays, corresponding to the output, input, and kernel data.
  
- **Threading**: The kernel is designed to use a grid of threads to handle the computation for each output pixel in parallel. Each thread is responsible for calculating a specific pixel in the output feature map.

```cpp
#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

