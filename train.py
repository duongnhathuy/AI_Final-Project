from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard,EarlyStopping
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.layers import Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorboard
import tensorflow as tf
import datetime
import matplotlib.pyplot
import os
import numpy as np
import pandas as pd
import sys

X = []
y = []
no_of_timesteps = 5


action = {}
pathSave = 'DATA'


for dirpath, dirnames, filenames in os.walk(pathSave):
    count=0
    for dirname in dirnames:
        print(dirname)
        action[f"{count}"] = dirname
        count+=1
        
print(action)

for key in action:
    txt_file =[]
    for file_name in os.listdir(f"{pathSave}\{action[key]}"):
        if file_name.endswith(".txt"):
            txt_file.append(os.path.join(f"{pathSave}\{action[key]}", file_name))
    
    
    for item in txt_file:
        data = pd.read_csv(item)
        dataset = data.iloc[:,1:].values
        n_sample = len(dataset)
        for i in range(no_of_timesteps, n_sample):
            X.append(dataset[i-no_of_timesteps:i,:])
            y.append(key)
        print(item)
    
    print(key)
    
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)
print(len(action))
#Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(no_of_timesteps,X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(len(action), activation='softmax'))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32,validation_data=(X_test, y_test))
model.save("AImodel.h5")

matplotlib.pyplot.plot(history.history['categorical_accuracy'])
matplotlib.pyplot.plot(history.history['val_categorical_accuracy'])
matplotlib.pyplot.title('model accuracy')
matplotlib.pyplot.ylabel('accuracy')
matplotlib.pyplot.xlabel('epochs')
matplotlib.pyplot.legend(['train','Validation'])
matplotlib.pyplot.show()