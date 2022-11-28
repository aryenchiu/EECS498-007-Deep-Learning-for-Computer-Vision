from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.nn.modules import Conv2d, Flatten
import torch.utils.data
from torch import device, flatten, nn, optim
from torch.autograd import Variable
from torch.nn import LeakyReLU, Linear, MaxPool2d, functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
NOISE_DIM = 96

def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
  """
  Generate a PyTorch Tensor of uniform random noise.

  Input:
  - batch_size: Integer giving the batch size of noise to generate.
  - noise_dim: Integer giving the dimension of noise to generate.
  
  Output:
  - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
    random noise in the range (-1, 1).
  """
  noise = None
  ##############################################################################
  # TODO: Implement sample_noise.                                              #
  ##############################################################################
  # Replace "pass" statement with your code
  noise = (1 - (-1)) * torch.rand((batch_size, noise_dim), dtype=dtype, device=device) + (-1)

  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################

  return noise



def discriminator():
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement discriminator.                                           #
  ############################################################################
  # Replace "pass" statement with your code
  model = nn.Sequential(
      nn.Linear(784, 256),
      nn.LeakyReLU(),
      nn.Linear(256, 256),
      nn.LeakyReLU(),
      nn.Linear(256, 1)
  )
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  
  return model


def generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement generator.                                               #
  ############################################################################
  # Replace "pass" statement with your code
  model = nn.Sequential(
      nn.Linear(noise_dim, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 784),
      nn.Tanh()
  )
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################

  return model  

def discriminator_loss(logits_real, logits_fake):
  """
  Computes the discriminator loss described above.
  
  Inputs:
  - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
  """
  loss = None
  ##############################################################################
  # TODO: Implement discriminator_loss.                                        #
  ##############################################################################
  # Replace "pass" statement with your code
  real_term = torch.nn.functional.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
  fake_term = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
  loss = real_term + fake_term
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss

def generator_loss(logits_fake):
  """
  Computes the generator loss described above.

  Inputs:
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing the (scalar) loss for the generator.
  """
  loss = None
  ##############################################################################
  # TODO: Implement generator_loss.                                            #
  ##############################################################################
  # Replace "pass" statement with your code
  loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss

def get_optimizer(model):
  """
  Construct and return an Adam optimizer for the model with learning rate 1e-3,
  beta1=0.5, and beta2=0.999.
  
  Input:
  - model: A PyTorch model that we want to optimize.
  
  Returns:
  - An Adam optimizer for the model with the desired hyperparameters.
  """
  optimizer = None
  ##############################################################################
  # TODO: Implement optimizer.                                                 #
  ##############################################################################
  # Replace "pass" statement with your code
  optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
  """
  Compute the Least-Squares GAN loss for the discriminator.
  
  Inputs:
  - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = None
  ##############################################################################
  # TODO: Implement ls_discriminator_loss.                                     #
  ##############################################################################
  # Replace "pass" statement with your code
  loss = 0.5 * torch.mean(torch.pow((scores_real - 1), 2)) + 0.5 * torch.mean(torch.pow(scores_fake, 2))
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss

def ls_generator_loss(scores_fake):
  """
  Computes the Least-Squares GAN loss for the generator.
  
  Inputs:
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = None
  ##############################################################################
  # TODO: Implement ls_generator_loss.                                         #
  ##############################################################################
  # Replace "pass" statement with your code
  loss = 0.5 * torch.mean(torch.pow((scores_fake - 1), 2))
  ##############################################################################
  #                              END OF YOUR CODE                              #
  ##############################################################################
  return loss


def build_dc_classifier():
  """
  Build and return a PyTorch nn.Sequential model for the DCGAN discriminator implementing
  the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement build_dc_classifier.                                     #
  ############################################################################
  # Replace "pass" statement with your code
  model = nn.Sequential(
      nn.Unflatten(1, (1, 28, 28)),
      nn.Conv2d(1, 32, 5, 1),  # [32, 24, 24]
      nn.LeakyReLU(),
      nn.MaxPool2d(2, 2),      # [32, 12, 12]
      nn.Conv2d(32, 64, 5, 1), # [64, 8 ,8]
      nn.LeakyReLU(),
      nn.MaxPool2d(2, 2),      # [64, 4, 4]
      nn.Flatten(),
      nn.Linear(64 * 4 * 4, 64 * 4 * 4),
      nn.LeakyReLU(),
      nn.Linear(64 * 4 * 4, 1)
  )
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################

  return model

def build_dc_generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the DCGAN generator using
  the architecture described in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement build_dc_generator.                                      #
  ############################################################################
  # Replace "pass" statement with your code
  model = nn.Sequential(
      nn.Linear(noise_dim, 1024),
      nn.ReLU(),
      nn.BatchNorm1d(1024),
      nn.Linear(1024, 7 * 7 * 128),
      nn.ReLU(),
      nn.BatchNorm1d(7 * 7 * 128),
      nn.Unflatten(1, (128, 7, 7)),
      nn.ConvTranspose2d(128, 64, 4, 2, 1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.ConvTranspose2d(64, 1, 4, 2, 1),
      nn.Tanh(),
      nn.Flatten()
  )
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################

  return model
