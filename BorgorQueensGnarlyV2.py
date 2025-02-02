
# coding: utf-8

# In[1]:


import tensorflow 

import pandas as pd
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")


# In[2]:


print("Tensorflow-version:", tensorflow.__version__)


# In[3]:


model_d=DenseNet121(weights='imagenet',include_top=False, input_shape=(128, 128, 3)) 

x=model_d.output

x= GlobalAveragePooling2D()(x)
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Dense(1024,activation='relu')(x) 
x= Dense(512,activation='relu')(x) 
x= BatchNormalization()(x)
x= Dropout(0.5)(x)

preds=Dense(3,activation='softmax')(x) #FC-layer #NOTE MANUAL CHANGE HERE


# model=Model(inputs=base_model.input,outputs=preds)
# model.summary()

# In[4]:


model=Model(inputs=model_d.input,outputs=preds)
model.summary()


# In[5]:


for layer in model.layers[:-8]:
    layer.trainable=False
    
for layer in model.layers[-8:]:
    layer.trainable=True


# In[6]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[7]:


data=[]
labels=[]
# TODO: random order
# random.seed(42)

train_csv = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')
train_csv['No Finding'] = train_csv['No Finding'].fillna(0.0)

for row in range(300):
    path = '/groups/CS156b/data/' + train_csv['Path'][row]
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    data.append(image)
    labels.append(train_csv['No Finding'][row])
   
print(data)
print(labels)


# In[8]:


data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)
print(labels[0])


# In[9]:


(xtrain,xtest,ytrain,ytest)=train_test_split(data,labels,test_size=0.4,random_state=42)
print(xtrain.shape, xtest.shape)


# In[10]:


anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)

datagen.fit(xtrain)

# print(xtrain)
# print(len(xtrain[0]))
# print(len(xtrain[0][0]))
# print(len(xtrain[0][0][0]))

# print(ytrain)


# Fits-the-model
history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=128),
               steps_per_epoch=xtrain.shape[0] //128,
               epochs=5, #MANUALLY EDITED
               verbose=2,
               callbacks=[anne, checkpoint],
               validation_data=(xtrain, ytrain))


# In[13]:


ypred = model.predict(xtest)

total = 0
accurate = 0
accurateindex = []
wrongindex = []

for i in range(len(ypred)):
    if np.argmax(ypred[i]) == np.argmax(ytest[i]):
        accurate += 1
        accurateindex.append(i)
    else:
        wrongindex.append(i)
        
    total += 1
    
print('Total-test-data;', total, '\taccurately-predicted-data:', accurate, '\t wrongly-predicted-data: ', total - accurate)
print('Accuracy:', round(accurate/total*100, 3), '%')


# In[ ]:


# FOR SUBMISSION
SUBMISSION_NAME = "submission3.csv"

test_data=[]
# TODO: random order
# random.seed(42)


submit_csv = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')

for row in range(len(submit_csv)):
    path = '/groups/CS156b/data/' + submit_csv['Path'][row]
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    test_data.append(image)
    
test_data = np.array(test_data, dtype="float32") / 255.0
print(test_data)
y_test = model.predict(test_data)
print(y_test)

submit_csv['No Finding'] = y_test

paths = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
for p in paths:
    submit_csv.insert(len(submit_csv.loc[0]), p, [1.0]*len(submit_csv))
del submit_csv['Path']

print(submit_csv)

submit_csv.to_csv(SUBMISSION_NAME, index=False)


# sol_csv = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')
# print(sol_csv)
# len(sol_csv.loc[0])
# paths = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
# for p in paths:
#     sol_csv.insert(len(sol_csv.loc[0]), p, [1.0]*len(sol_csv))
# del sol_csv['Path']
# sol_csv

# sol_csv.to_csv('submission2.csv', index=False)





# In[ ]:


ytest = model.predict(test_data)
print(ytest)


# In[26]:


label = ['-1', '0', '1']
imidx = random.sample(wrongindex, k=9) # set 'wrongindex' or 'accurateindex'

nrows = 3
ncols = 3
fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))

n = 0
for row in range(nrows):
    for col in range(ncols):
            ax[row,col].imshow(xtest[imidx[n]])
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(label[np.argmax(ypred[imidx[n]])], label[np.argmax(ytest[imidx[n]])]))
            n += 1

plt.show()


# In[15]:


Ypred = model.predict(xtest)

Ypred = np.argmax(Ypred, axis=1)
Ytrue = np.argmax(ytest, axis=1)

cm = confusion_matrix(Ytrue, Ypred)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap="rocket_r", fmt=".01f",annot_kws={'size':16}, annot=True, square=True, xticklabels=label, yticklabels=label)
ax.set_ylabel('Actual', fontsize=20)
ax.set_xlabel('Predicted', fontsize=20)

