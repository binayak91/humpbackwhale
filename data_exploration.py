###Start of Code##

##Importing dependencies##

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.utils.data, torchvision
import PIL

import os.path
import time
import skimage, skimage.io
import copy

# Load the data and visualize it as a histogram
df = pandas.read_csv('C:/Users/Bikash/Downloads/train.csv')
df.head(n=10)

unique = pandas.value_counts(df.Id)

num_classes = unique.values.shape[0]
num_classes
plt.figure()
plt.plot(range(1,num_classes),unique.values[1:],'-k')
plt.xlabel('labels converted to arbitray int')
plt.ylabel('occurences')
plt.show()