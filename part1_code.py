import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time


def positional_encoding(x, num_frequencies=6, incl_input=True):
    """
    Apply positional encoding to the input.
    
    Args:
    x (torch.Tensor): Input tensor to be positionally encoded. 
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the 
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor. 
    """
    results = []
    # D = x.shape[-1]
    if incl_input:
        results.append(x)
    for i in range(num_frequencies):
        # for j in range(D):
        sin = torch.sin((2 ** i) * torch.pi * x)
        cos = torch.cos((2 ** i) * torch.pi * x)

        results.append(sin)
        results.append(cos)

    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1)


class model_2d(nn.Module):
    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """

    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        #############################  TODO 1(b) BEGIN  ############################
        input_dimension = 2 + 2 * num_frequencies * 2
        # Output_dimension = 3
        self.layer1 = nn.Linear(input_dimension, filter_size)
        self.layer2 = nn.Linear(filter_size, filter_size)
        self.layer3 = nn.Linear(filter_size, 3)

        # self.Relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    #############################  TODO 1(b) END  ##############################

    def forward(self, x):
        #############################  TODO 1(b) BEGIN  ############################

        print("Input shape: ", x.shape)
        x = x.view(x.shape[0], -1)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.sigmoid(self.layer3(x))

        #############################  TODO 1(b) END  ##############################
        return x


def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding,
                   show=True):
    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000

    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)

    #############################  TODO 1(c) BEGIN  ############################
    # Define the optimizer
    optimizer = torch.optim.Adam(model2d.parameters(), lr=lr)
    #############################  TODO 1(c) END  ############################

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them

    # w=width_painting
    # h=height_painting
    # w=width
    # h=height

    Width = torch.linspace(0, 1, width)
    Height = torch.linspace(0, 1, height)

    # no_of_cells=w*h
    x = torch.zeros((width * height, 2))

    for i in range(height):
        for j in range(width):
            x[(i * width) + j] = torch.tensor([Width[j], Height[i]])  ##doubt

    Positional_Encoding = positional_encoding(x, num_frequencies=num_frequencies)
    Positional_Encoding = Positional_Encoding.to(device)
    #############################  TODO 1(c) END  ############################

    for i in range(iterations + 1):
        optimizer.zero_grad()
        #############################  TODO 1(c) BEGIN  ############################
        # Run one iteration
        pred = model2d.forward(x=Positional_Encoding).reshape((height, width, 3))
        loss = F.mse_loss(pred, test_img)
        loss.backward()
        optimizer.step()  # Used to update parameters
        # Compute mean-squared error between the predicted and target images. Backprop!

        #############################  TODO 1(c) END  ############################

        # Display images/plots/stats
        if i % display == 0 and show:
            #############################  TODO 1(c) BEGIN  ############################
            # Calculate psnr

            psnr = 10 * torch.log10(((pred.max()) ** 2) / loss)

            #############################  TODO 1(c) END  ############################

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                  "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

            # if i==iterations:
            #  np.save('result_'+str(num_frequencies)+'.npz',pred.detach().cpu().numpy())

    print('Done!')
    return pred.detach().cpu()
