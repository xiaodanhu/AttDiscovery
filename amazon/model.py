import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# defining the model architecture
'''
class Net_material(nn.Module):   
  def __init__(self):
      super(Net_material, self).__init__()
      self.linear_layers_color = nn.Sequential(
          nn.Linear(1024, 512),
          nn.Linear(512, 13)
      )
      self.linear_layers_shape = nn.Sequential(
          nn.Linear(1024, 512),
          nn.Linear(512, 7)
      )
      self.linear_layers_material = nn.Sequential(
          nn.Linear(1024, 512),
          nn.Linear(512, 7)
      )  

  def forward(self, x):
      out_color = self.linear_layers_color(x)
      out_shape = self.linear_layers_shape(x)
      out_material = self.linear_layers_material(x)
      return [out_color, out_shape, out_material]
'''

class Net(nn.Module):   
  def __init__(self, dimc, dims):
      super(Net, self).__init__()
      self.encode = nn.Sequential(
          nn.Linear(768, 512),
          nn.Linear(512, 256)
      )

      self.color = nn.Sequential(
          nn.Linear(256, 128),
          nn.Linear(128, dimc)
      )

      self.shape = nn.Sequential(
          nn.Linear(256, 128),
          nn.Linear(128, dims)
      )

  def forward(self, x):
        x = self.encode(x)
        yc = self.color(x)
        ys = self.shape(x)
        return [yc, ys]