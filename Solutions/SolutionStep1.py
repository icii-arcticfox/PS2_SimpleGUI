from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#>1#(13390530264536):(13390530264536)
from tensorflow.keras.losses import sparse_categorical_crossentropy
#<1#(13390530264536)~%(-1952786418) #>1#(13390593024730):(13390593024730)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
#<1#(13390593024730)~%(-1092476421) #>1#(13390554419590):(13390554419590)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import tensorflow as tf
#<1#(13390554419590)~%(-1919629388)
from tkinter import *

#[Data mnist]#@54419590 #>1#(01390554419590):(01390554419590)
(mnist_train_images, mnist_training_labels), (mnist_validation_images, mnist_test_labels) = mnist.load_data()

mnist_train_images = mnist_train_images.reshape((60000, 28, 28, 1))
mnist_validation_images = mnist_validation_images.reshape((10000, 28, 28, 1))

mnist_training_labels = to_categorical(mnist_training_labels)
mnist_test_labels = to_categorical(mnist_test_labels)
#<1#(01390554419590)~%(-372442558)

#[NeuralNetwork --denseLayers 15 --denseStart 64 --denseEnd 16]#@93024730 #>1#(01390593024730):(01390593024730)
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
#<1#(01390593024730)~%(1863931219)

#[Train --epochs 5]#@30264536 #>1#(01390530264536):(01390530264536)
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
#<1#(01390530264536)~%(1787826421)


window = Tk()
window.title( 'Training Plots' )
window.geometry("800x600")

window.mainloop()