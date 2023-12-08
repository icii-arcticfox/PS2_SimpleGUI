#>1#(13390513717447):(13390513717447)
from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#<1#(13390513717447)~%(-454843538) #>1#(13390574329790):(13390574329790)
from tensorflow.keras.losses import sparse_categorical_crossentropy
#<1#(13390574329790)~%(-1952786418) #>1#(13390519470345):(13390519470345)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
#<1#(13390519470345)~%(-1092476421) #>1#(13390582270238):(13390582270238)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import tensorflow as tf
#<1#(13390582270238)~%(-1919629388)

#[Data mnist]#@82270238 #>1#(01390582270238):(01390582270238)
(mnist_train_images, mnist_training_labels), (mnist_validation_images, mnist_test_labels) = mnist.load_data()

mnist_train_images = mnist_train_images.reshape((60000, 28, 28, 1))
mnist_validation_images = mnist_validation_images.reshape((10000, 28, 28, 1))

mnist_training_labels = to_categorical(mnist_training_labels)
mnist_test_labels = to_categorical(mnist_test_labels)
#<1#(01390582270238)~%(-372442558)

#[NeuralNetwork --denseLayers 15 --denseStart 64 --denseEnd 16]#@19470345 #>1#(01390519470345):(01390519470345)
# Create the model
model = Sequential()

model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(57, activation='relu'))
model.add(Dense(52, activation='relu'))
model.add(Dense(47, activation='relu'))
model.add(Dense(43, activation='relu'))
model.add(Dense(39, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))


# Compile the model
model.compile( 
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
    )


model.summary()
#<1#(01390519470345)~%(1863931219)

#[Train --epochs 5]#@74329790 #>1#(01390574329790):(01390574329790)
# Fit data to model
history = model.fit(
    mnist_train_images,
    mnist_training_labels,
    batch_size=128,
    epochs=5,
    verbose=1,
    validation_split=0.2
)


# Generate generalization metrics
score = model.evaluate(mnist_validation_images, mnist_test_labels, verbose=0, batch_size=1)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
#<1#(01390574329790)~%(1787826421)

#[GUI]#@13717447 #>1#(01390513717447):(01390513717447)
window = Tk()
window.title('')
window.geometry("800x600")
#<1#(01390513717447)~%(-1671105959) #>1#(23390513717447):(23390513717447)
window.mainloop()
#<1#(23390513717447)~%(-532138404)