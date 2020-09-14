

# import library
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout
from keras.models import model_from_json


# load data
data=pd.read_csv('/content/drive/My Drive/handwritten_data_785.csv.zip')


# get label
l=data['0']
data.drop('0',axis=1,inplace=True)


# preprocess input data
input_data=StandardScaler().fit_transform(data)
input_data=np.array(input_data).reshape(372037,28,28,1)


# convert label to categorical data
label=to_categorical(l)


# make model_cnn
model_cnn=Sequential()

model_cnn.add(Conv2D(kernel_size=(3,3),filters=32,padding="valid",input_shape=(28,28,1),activation='relu'))
model_cnn.add(MaxPooling2D())
model_cnn.add(BatchNormalization())


model_cnn.add(Conv2D(kernel_size=(3,3),filters=64,padding="same",activation='relu'))
model_cnn.add(MaxPooling2D())
model_cnn.add(BatchNormalization())


model_cnn.add(Conv2D(kernel_size=(3,3),filters=128,padding="same",activation='relu'))
model_cnn.add(MaxPooling2D())
model_cnn.add(BatchNormalization())


model_cnn.add(Conv2D(kernel_size=(3,3),filters=256,padding="same",activation='relu'))
model_cnn.add(MaxPooling2D())
model_cnn.add(BatchNormalization())

model_cnn.add(Flatten())

model_cnn.add(Dense(750,activation='relu'))
model_cnn.add(Dense(800,activation='relu'))
model_cnn.add(Dense(26,activation='sigmoid'))

# compile model_cnn
model_cnn.compile(optimizer='RMSprop',loss='binary_crossentropy',metrics=['accuracy'])


# fit model_cnn
model_cnn.fit(input_data,label,batch_size=256,validation_split=0.1,epochs=20)


# serialize model to JSON
model_json = model_cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model_cnn.save_weights("model.h5")
print("Saved model to disk")
 




