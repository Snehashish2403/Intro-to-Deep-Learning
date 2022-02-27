#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random


# In[10]:


np.random.seed(0)


# In[11]:


#Load the mnist dataset
(X_train, Y_train),(X_test,Y_test)=mnist.load_data()


# In[12]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[13]:


assert(X_train.shape[0]==Y_train.shape[0]),"There are unequal number of Train images and Train Labels!"
assert(X_test.shape[0]==Y_test.shape[0]),"There are unequal number of Test images and Test Labels!"
assert(X_train.shape[1:]==(28,28)),"Dimensions of images are not 28X28!"
assert(X_test.shape[1:]==(28,28)),"Dimensions of images are not 28X28!"


# In[23]:


#Visualise the data we have
num_samples=[]

cols=5
num_classes=10
fig,axs=plt.subplots(nrows=num_classes,ncols=cols,figsize=(5,10))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected=X_train[Y_train==j]
        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected)-1),: ,:], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i==2:
            axs[j][i].set_title(str(j))
            num_samples.append(len(x_selected))


# In[26]:


plt.figure(figsize=(12,4))
plt.bar(range(0,num_classes),num_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of Images")


# In[27]:


#One hot encode the labels of both train and test data
Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)


# In[28]:


#Normalising the data
X_train=X_train/255
X_test=X_test/255

#What we have done here is that we have reduced the range of the color intensity of each pixel from (0-255) to (0-1)
#Thus reducing a lot of computational cost and still preserving the features


# In[31]:


#We now flatten this image array so that we can perform matrix multiplications
num_pixels=28*28
X_train=X_train.reshape(X_train.shape[0],num_pixels)
print(X_train.shape)
X_test=X_test.reshape(X_test.shape[0],num_pixels)
print(X_test.shape)


# In[ ]:




