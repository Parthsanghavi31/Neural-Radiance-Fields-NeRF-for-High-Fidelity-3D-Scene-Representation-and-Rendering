# Neural-Radiance-Fields-NeRF-for-High-Fidelity-3D-Scene-Representation-and-Rendering

Neural Radiance Fields (NeRF) is a technique that uses deep learning to represent a 3D scene in a continuous, volumetric manner. This repository contains a detailed implementation of NeRF, enabling 3D scene reconstruction from 2D images.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Methods and Functions](#methods-and-functions)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

NeRF leverages the capabilities of neural networks to represent a scene using a continuous volumetric scene function. This function is then used to render novel views of the scene, eliminating the need for traditional 3D representation. The strength of NeRF lies in its ability to produce high-fidelity results with minimal artifacts compared to other 3D reconstruction techniques.

## Dependencies

- numpy
- torch
- matplotlib
- gdown

Ensure you have a CUDA-compatible GPU for efficient computation.

## Methods and Functions

### Positional Encoding

The `positional_encoding` function is used to apply a positional encoding to the input tensor. This encoding enhances the capacity of the network to represent high-frequency functions without increasing the number of network parameters.

### 2D Model

The `model_2d` class defines a 2D neural network model comprising three fully connected layers with ReLU activations and a final sigmoid activation. This model is trained to predict RGB values for each 2D coordinate.

### Training the 2D Model

The `train_2d_model` function is responsible for training the 2D model. It uses the Adam optimizer and Mean Squared Error (MSE) loss. The function also computes the Peak Signal-to-Noise Ratio (PSNR) to evaluate the quality of the reconstructed images.

### Data Loading and Preprocessing

The dataset used is 'lego_data', which contains images, camera poses, and intrinsic parameters. The data is loaded into the environment using `gdown` and then processed to be fed into the model.

### NeRF Model Training

The main training loop trains the NeRF model using the provided dataset. During each iteration, a random image is selected, and the model is trained to minimize the difference between the predicted and actual images.

## Usage

1. **Setup**:
   Clone the repository and install the necessary dependencies.

2. **Data**:
   Download the 'lego_data' dataset using the provided `gdown` link. This dataset contains images, camera poses, and intrinsic parameters.

3. **Training**:
   Execute the main training loop. This will train the NeRF model on the dataset and produce reconstructed 3D scenes.

4. **Visualization**:
   Post-training, visualize the reconstructed scenes and compare them with the original images.
## Results

The NeRF model's capability is showcased through the novel view reconstruction of the lego dataset. Below is the visual representation of the reconstructed scene alongside the original image for comparison.

![Original Image](https://github.com/Parthsanghavi31/Neural-Radiance-Fields-NeRF-for-High-Fidelity-3D-Scene-Representation-and-Rendering/blob/main/Nerf_algorithm.png)

*NeRF algorithm working*

![Novel View](https://github.com/Parthsanghavi31/Neural-Radiance-Fields-NeRF-for-High-Fidelity-3D-Scene-Representation-and-Rendering/blob/main/3D_Test_image_PSNR.png)
*Novel View Reconstruction*

The Peak Signal-to-Noise Ratio (PSNR) metric, which quantifies the quality of the reconstructed image compared to the original, is computed for the novel view. The obtained PSNR score for the scene is: **25.45 dB**.


## Acknowledgements

This implementation draws inspiration from the original NeRF paper by [Ben Mildenhall et al.](https://arxiv.org/abs/2003.08934). We extend our gratitude to the authors for their pioneering work and for making their research accessible to the public.

---

For issues or suggestions related to this implementation, please raise an issue or submit a pull request.

