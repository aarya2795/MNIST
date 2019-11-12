#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D


# In[2]:


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


fashion_mnist = tf.keras.datasets.fashion_mnist


# In[4]:


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
print("x_train shape:", train_images.shape, "labels train shape:", train_labels.shape)


# In[5]:


test_images = test_images / 255.0
ntrain = train_images[10000:30000]
ntlabels = train_labels[10000:30000]


# In[6]:


ntrain = ntrain.reshape(20000, 28, 28, 1)
print(ntrain.shape)


# In[7]:


test_images = test_images.reshape(10000, 28, 28, 1)
print(test_images.shape)


# In[8]:


model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(5408, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


# In[9]:


model.summary()


# In[13]:


model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'sparse_categorical_crossentropy',
              metrics = [metrics.sparse_categorical_accuracy])


# In[14]:


model.fit(ntrain, ntlabels, epochs=10)


# In[15]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[16]:


print('Test accuracy:', test_acc)

