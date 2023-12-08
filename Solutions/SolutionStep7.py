#>1#(13390513436986):(13390513436986)
import numpy as np
import matplotlib.pyplot as plt
#<1#(13390513436986)~%(-27499987) #>1#(13390568160469):(13390568160469)
from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#<1#(13390568160469)~%(-454843538) #>1#(13390555143018):(13390555143018)
from tensorflow.keras.losses import sparse_categorical_crossentropy
#<1#(13390555143018)~%(-1952786418) #>1#(13390551770568):(13390551770568)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
#<1#(13390551770568)~%(-1092476421) #>1#(13390543221947):(13390543221947)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import tensorflow as tf
#<1#(13390543221947)~%(-1919629388)

#[Data mnist]#@43221947 #>1#(01390543221947):(01390543221947)
(mnist_train_images, mnist_training_labels), (mnist_validation_images, mnist_test_labels) = mnist.load_data()

mnist_train_images = mnist_train_images.reshape((60000, 28, 28, 1))
mnist_validation_images = mnist_validation_images.reshape((10000, 28, 28, 1))

mnist_training_labels = to_categorical(mnist_training_labels)
mnist_test_labels = to_categorical(mnist_test_labels)
#<1#(01390543221947)~%(-372442558)

#[NeuralNetwork --denseLayers 15 --denseStart 64 --denseEnd 16]#@51770568 #>1#(01390551770568):(01390551770568)
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
#<1#(01390551770568)~%(1863931219)

#[Train --epochs 5]#@55143018 #>1#(01390555143018):(01390555143018)
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
#<1#(01390555143018)~%(1787826421)

#[GUI 'Training Plots']#@68160469 #>1#(01390568160469):(01390568160469)
window = Tk()
window.title('Training Plots')
window.geometry("800x600")
#<1#(01390568160469)~%(522815045)

#[Visualize --sampleImage]#@13436986
#[AddPlotToWindow]#@64978611
#[GUIButton]#@33117307
def showTrainingImage():
#>1#(04390564978611):(04390564978611)
    plt.figure()
#<1#(04390564978611)~%(1347837467) #>1#(03390513436986):(03390513436986)
    #***Visualize sample training images
    img = mnist_validation_images[0]
    label = mnist_test_labels[0]
    print('Actual Label: ' + str(np.argmax(label)))
    plt.imshow(img, cmap='gray')
    # plt.show()
#<1#(03390513436986)~%(-2076572407) #>1#(03390564978611):(03390564978611)
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, columnspan=window.grid_size()[0], pady=10)
#<1#(03390564978611)~%(1036401314) #>1#(21390533117307):(21390533117307)
plot_button = Button(master = window,
    command = showTrainingImage,
    height = 2,
    width = 21,
    text = 'Show Training Image' )
plot_button.grid(row=0, column=0, padx=5)
window.columnconfigure( 0, weight=1)
#<1#(21390533117307)~%(1764756756) #>1#(23390568160469):(23390568160469)
window.mainloop()
#<1#(23390568160469)~%(-532138404)