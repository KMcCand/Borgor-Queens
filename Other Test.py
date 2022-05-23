
# coding: utf-8

# In[189]:


IMG_SIZE = 128
BATCH = 32
SEED = 42


# In[267]:


import pandas as pd       
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dropout
from tensorflow.keras.applications import DenseNet121


from numpy.random import seed
import copy
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
from tensorflow.keras import callbacks, layers
import tensorflow as tf
seed(42)
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import cv2
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# In[264]:


data=[]
labels=[]

category = "Pneumonia"

train_csv = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')
train_csv[category] = train_csv[category].fillna(0.0)

for row in range(int(len(train_csv)/50)):
    path = '/groups/CS156b/data/' + train_csv['Path'][row]
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    data.append(image)
    labels.append(train_csv[category][row])
    
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
mlb = LabelBinarizer()
labels = mlb.fit_transform(labels)

(xtrain,xtest,ytrain,ytest)=train_test_split(data,labels,test_size=0.4,random_state=42)



datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)

datagen.fit(xtrain)


# In[265]:


maptrain= dict()
maptrain["trainx"] = list(xtrain)
maptrain["trainy"] = list(ytrain)

maptest = dict()
maptest["testx"] = list(xtest)
maptest["testy"] = list(ytest)

train_df = pd.DataFrame(maptrain)
test_df = pd.DataFrame(maptest)

train_df


# In[269]:


# Model from a website
# def get_model():
#     cnn = Sequential()
#     cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
#     cnn.add(MaxPooling2D(pool_size = (2, 2)))
    
#     cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
#     cnn.add(MaxPooling2D(pool_size = (2, 2)))
    
#     cnn.add(Conv2D(128, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
#     cnn.add(MaxPooling2D(pool_size = (2, 2)))

#     cnn.add(Dense(activation = 'relu', units = 128))
#     cnn.add(Dense(activation = 'relu', units = 64))

#     cnn.add(Dense(1, activation='softmax'))
    
#     return

# def get_model():
    
    #Input shape = [width, height, color channels]
#     x = layers.Input(shape=(128, 128, 3))
    
    
#     Another model from a website START
#     # Block One
#     x = layers.Conv2D(filters=32, kernel_size=3, padding='valid')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.MaxPool2D()(x)
#     x = layers.Dropout(0.2)(x)
    
#     # Block Two
#     x = layers.Conv2D(filters=64, kernel_size=3, padding='valid')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.MaxPool2D()(x)
#     x = layers.Dropout(0.2)(x)
    
#     # Block Three
#     x = layers.Conv2D(filters=128, kernel_size=3, padding='valid')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.MaxPool2D()(x)
#     x = layers.Dropout(0.4)(x)
# Another model END

# #     Emily model START
#     x = GlobalAveragePooling2D()(x)
#     x = BatchNormalization()(x)
  
#     x = Dropout(0.5)(x)
#     x = Dense(1024,activation='relu')(x) 
#     x = Dense(512,activation='relu')(x) 
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
# #     Emily model END

    
#     # Head
#     #x = layers.BatchNormalization()(x)
#     x = layers.Flatten()(x)
# #     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
    
#     #Final Layer (Output)
#     output = layers.Dense(3, activation='softmax')(x)
    
#     model = keras.Model(inputs=[x], outputs=output)
#     model.compile()

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


model=Model(inputs=model_d.input,outputs=preds)
model.summary()


# In[272]:


early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
callbacks_list = [early, learning_rate_reduction]

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()




# In[274]:


anne = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)

datagen.fit(xtrain)

history = model.fit(datagen.flow(xtrain, ytrain, batch_size=128),
               steps_per_epoch=xtrain.shape[0] // 128,
               epochs=10, #MANUALLY EDITED
               verbose=1,
               callbacks=[anne, checkpoint],
               validation_data=(xtest, ytest))


# In[278]:


test_data=[]

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

submit_csv[category] = y_test

paths = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
for p in paths:
    submit_csv.insert(len(submit_csv.loc[0]), p, [1.0]*len(submit_csv))
del submit_csv['Path']

print(submit_csv)

submit_csv.to_csv("Submission_test_pneumonia", index=False)


# In[1]:


submit_csv


# In[ ]:


# End here


# In[104]:


img_height = 500
img_width = 500

train_path = '/groups/CS156b/data/'
test_path = '/groups/CS156b/data/'


train_csv = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')

m1data = train_csv[["Path", category]]

replaced = m1data.fillna(0.0)


# In[110]:


def append_ext(fn):
    return test_path + fn


# In[111]:


replaced["Path"] = append_ext(replaced["Path"])


# In[112]:


replaced


# In[114]:


train_df, val_df = train_test_split(replaced, test_size = 0.20, random_state = SEED, stratify = replaced[category])


# In[115]:


train_df


# In[116]:


val_df


# In[174]:


train_datagen = ImageDataGenerator(rescale=1/255.,
                                  zoom_range = 0.1,
                                  #rotation_range = 0.1,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1)

val_datagen = ImageDataGenerator(rescale=1/255.)

ds_train = train_datagen.flow_from_dataframe(train_df[:int(len(train_df)/50)],
                                             #directory=train_path, #dataframe contains the full paths
                                             x_col = 'Path',
                                             y_col = category,
                                             target_size = (IMG_SIZE, IMG_SIZE),
                                             class_mode = 'raw',
                                             batch_size = BATCH,
                                             seed = SEED,
                                             validate_filenames = False)

ds_val = val_datagen.flow_from_dataframe(val_df[:int(len(val_df)/50)],
                                            #directory=train_path,
                                            x_col = 'Path',
                                            y_col = category,
                                            target_size = (IMG_SIZE, IMG_SIZE),
                                            class_mode = 'raw',
                                            batch_size = BATCH,
                                            seed = SEED,
                                            validate_filenames = False)


# In[175]:


early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-7,
    restore_best_weights=True,
)

plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor = 0.2,                                     
    patience = 2,                                   
    min_delt = 1e-7,                                
    cooldown = 0,                               
    verbose = 1
)


# In[178]:


early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
callbacks_list = [early, learning_rate_reduction]

history = model.fit(xtrain,
          batch_size = BATCH, epochs = ++10,
          validation_data=ds_val,
          callbacks=callbacks_list,
          steps_per_epoch=(int(len(train_df)/50)/BATCH),
          validation_steps=(int(len(val_df)/50)/BATCH));

# model.fit(ds_train, epochs=25, validation_data=ds_val, callbacks=callbacks_list)

# model.fit(ds_train)


# In[99]:


model.save("my_model_test")


# In[121]:


model = keras.models.load_model("my_model_test")


# In[122]:


sol_csv = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')
# len(sol_csv.loc[0])
# paths = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

sol_csv
# df_test = pd.DataFrame(np.concatenate([['Normal']*len(test_normal) , ['Pneumonia']*len(test_pneumonia)]), columns = ['class'])


# In[123]:


sol_csv["Path"] = append_ext(sol_csv["Path"])


# In[124]:


sol_csv


# In[125]:


test_datagen = ImageDataGenerator(rescale=1/255.)

ds_test = test_datagen.flow_from_dataframe(sol_csv,
                                            x_col = 'Path',
                                            y_col = None,
                                            target_size = (IMG_SIZE, IMG_SIZE),
                                            class_mode = None,
                                            batch_size = 1,
                                            seed = None,
                                            validate_filenames = False)


# In[129]:


ds_test.reset()
predictions = model.predict(ds_test, steps=len(ds_test), verbose=1)


# In[131]:


predictions


# In[100]:


pred_list = []
for i in range(len(predictions)):
    pred_list.append(predictions[i][0])


# In[102]:


sol_csv = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')
sol_csv.insert(len(sol_csv.loc[0]), category, pred_list)

paths = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
for p in paths:
    sol_csv.insert(len(sol_csv.loc[0]), p, [0.0]*len(sol_csv))
del sol_csv['Path']


# In[103]:


sol_csv


# In[104]:


sol_csv.to_csv('submission5.csv', index=False)


# In[ ]:


score = model.evaluate(ds_val, steps = len(val_df)/BATCH, verbose = 0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])


# In[ ]:


score = model.evaluate(ds_test, steps = len(df_test), verbose = 0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

