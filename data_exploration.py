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

#the most common aspect ratio
H = []
W = []
count = 0
folder = 'C:/Users/Bikash/Downloads/train/'
t0 = time.time()
for fname in os.listdir(folder):
    if fname[-3:] == 'jpg':
        img = skimage.io.imread(folder+fname)
        H.append(img.shape[0])
        W.append(img.shape[1])
        count += 1
#     if count == 100:
#         break

H = np.array(H,dtype='float')
W = np.array(W,dtype='float')
A = H/W
unique, ret_counts = np.unique(A,return_counts=True)

plt.figure()
plt.plot(unique,ret_counts)
plt.xlabel('Aspect ratio')
plt.ylabel('Occurrences')