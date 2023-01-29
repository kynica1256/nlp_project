from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SimpleRNN
from keras.layers import LSTM


import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import preprocessing

from method_text_encode import  method_text_encode

label_encoder = preprocessing.LabelEncoder()

np.random.seed(1)

methods = method_text_encode()
'''
label_encoder = preprocessing.LabelEncoder()
token_words = label_encoder.fit_transform(methods.all_words)
print(token_words)
'''

methods.encode_main()
train_x = methods.train_x
train_y = methods.train_y
maxcount = methods.MaxCount
maxseq = methods.MaxSeq
matrixwords = methods.matrix_words

pred_data = methods.text_in_matrix(["Привет как дела ?", "у меня нормально", "пока"])

print(pred_data[0])

print(train_x[0])



#train_x = np.delete(train_x, 0, axis=1)
#train_y = np.delete(train_y, 0, axis=1)




#train_x = np.random.random([32, 10, 8]).astype(np.float32)
#train_y = np.random.random([32, 10, 10]).astype(np.float32)


#print(train_x)
#print(train_y)

#print(len(np.random.random([32, 10, 8]).astype(np.float32)[0][0]))
#train_x = 0.2*np.random.random((10,10)) - 0.1

#train_y = 0.2*np.random.random((10,10)) - 0.1

#sys.exit()


model = keras.Sequential()

model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(maxseq, maxcount)))
#model.add(SimpleRNN(38, return_sequences=True, activation='relu'))
model.add(Dense(10, activation='relu'))
#model.add(Dense(50, activation='tanh', input_shape=(100,)))
#model.add(Dense(10, activation='sigmoid', input_shape=(10,)))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_x, train_y, epochs=100, batch_size=30, verbose=2)
print(model.summary())




#data_chat_ = ["привет", "как дела ?", "пока"]

#pred = model.predict(np.array([matrix_words[i] for i in range(MaxCount) if all_words[i] == "привет"]))
#print(pred)
print(pred_data)
pred = model.predict(pred_data)
print(pred)

print(matrixwords)

#print("\n\n")
#print(matrix_words)
#plt.plot(history.history['loss'])
#plt.grid(True)
#plt.show()
