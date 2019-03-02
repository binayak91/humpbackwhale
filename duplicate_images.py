import matplotlib.pyplot as plt
import numpy as np
import pandas
import PIL
import imagehash
import time

df_train = pandas.read_csv('C:/Users/Bikash/Downloads/data/train.csv')

# Create a new column with the hash value for the image
t0=time.time()
img_hash = df_train.Image.apply(lambda x: imagehash.phash(PIL.Image.open('C:/Users/Bikash/Downloads/data/train/'+x)))
df_train['Hash_val'] = img_hash
t1=time.time()
t2=t1-t0

num_duplicates = (df_train.Hash_val.value_counts() > 1).sum()
num_duplicates