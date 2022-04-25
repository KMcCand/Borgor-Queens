#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

train_csv = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')
# for row in range(1, 10):
#     path = '/groups/CS156b/data/' + train_csv['Path'][row]
#     image = mpimg.imread(path)
#     res = resize(im, (150, 54))
#     print(train_csv['Path'][row])
#     plt.imshow(res)
#     plt.show()


# In[2]:


sol_csv = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')
len(sol_csv.loc[0])
paths = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
for p in paths:
    sol_csv.insert(len(sol_csv.loc[0]), p, [1.0]*len(sol_csv))
del sol_csv['Path']
sol_csv

sol_csv.to_csv('submission2.csv', index=False)


# In[ ]:





# In[15]:


#Some Basic Imports
import matplotlib.pyplot as plt #For Visualization
import numpy as np              #For handling arrays
import pandas as pd             # For handling data
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from tensorflow import *
#Define Directories for train, test & Validation Set
train_path = '/groups/CS156b/data/student_labels'
test_path = '/groups/CS156b/data/student_labels'
# valid_path = '/content/dataset/cnn/pneumonia_revamped/val'

#Define some often used standard parameters
#The batch refers to the number of training examples utilized in one #iteration
batch_size = 16
#The dimension of the images we are going to define is 300x300 img_height = 300
img_height = 300
img_width = 300


# In[16]:


# Create Image Data Generator for Train Set
image_gen = ImageDataGenerator(
                                  rescale = 1./255
#                                   shear_range = 0.2,
#                                   zoom_range = 0.2,
#                                   horizontal_flip = True,          
                               )
# Create Image Data Generator for Test/Validation Set
test_data_gen = ImageDataGenerator(rescale = 1./255)


# In[69]:


from PIL import Image
# import cv2
labels = []
imgs = []
for row in range(300):
    path = '/groups/CS156b/data/' + train_csv['Path'][row]
    img = Image.open(path)
    img = img.resize((300, 300))
    
    imgs.append(np.array(img))
    labels.append([train_csv['No Finding'][row]])
    

labels = np.asarray(labels)

val_labels = np.asarray(labels[0:50])
val_imgs = np.asarray(imgs[0:50])
train_labels = np.asarray(labels[50:])
train_imgs = np.asarray(imgs[50:])


# In[85]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau


# In[86]:


cnn = Sequential()
# cnn.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (300, 300, 1)))
# cnn.add(MaxPooling2D(pool_size = (2, 2)))
# cnn.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = (300, 300, 1)))
# cnn.add(MaxPooling2D(pool_size = (2, 2)))
# cnn.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = (300, 300, 1)))


cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
# cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
# cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'relu', units = 64))
cnn.add(Dense(activation = 'sigmoid', units = 1))
cnn.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])


# In[87]:


x_train = np.stack(imgs, axis = 0)
history = cnn.fit(x_train, labels, epochs = 10, validation_data = (val_imgs, val_labels))


# In[67]:


from PIL import Image
sol_csv = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')
sol_labels = []
sol_imgs = []
for row in range(10):
    path = '/groups/CS156b/data/' + sol_csv['Path'][row]
    img = Image.open(path)
    img = img.resize((300, 300))
    
    sol_imgs.append(np.array(img))
#     sol_labels.append([train_csv['No Finding'][row]])
    
sol_imgs = np.asarray(sol_imgs)

predictions = cnn.predict(sol_imgs)


# In[68]:


predictions


# In[ ]:




