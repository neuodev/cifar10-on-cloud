import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
import json
import os

print('Load Data...')
(X_train, y_train), (X_test, y_test) = load_data()
print('Data', X_train.shape)

print('Build the model..')
model = Sequential()
model.add(Conv2D(filters=3, kernel_size=2, input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=3, kernel_size=2))
model.add(MaxPooling2D(2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=3, kernel_size=2))
model.add(MaxPooling2D(2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy, 
    optimizer='adam', 
    metrics=['accuracy']
)

print(model.summary())

print('Start Traning..')
epochs = None
batch_size = None
while not epochs:
    epcohs_input = input('Epochs: \n')
    if epcohs_input:
        epochs = int(epcohs_input)

while not batch_size:
    batch_input = input('Batch Size: \n')
    if batch_input:
        batch_size = int(batch_input)


history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=epochs, batch_size=batch_size, 
    verbose=1
).history

# write the data to json file 
LOGS = './logs'
if not os.path.isdir(LOGS):
    os.mkdir(LOGS)

log_file = os.path.join(LOGS, f"cifar10-{epochs}-{batch_size}.json")
with open(log_file, 'w') as f:
    f.write(json.dumps(history))