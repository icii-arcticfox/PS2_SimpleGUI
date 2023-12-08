#>1#(13390563557867):(13390563557867)
import scikitplot as skplt
#<1#(13390563557867)~%(-1087051367) #>1#(13390513436986):(13390513436986)
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
#<1#(21390533117307)~%(1764756756)


#[Histogram mnist_train_images]#@95280198
#[AddPlotToWindow]#@24484340
#[GUIButton]#@62142525
def pixelHistogram():
#>1#(04390524484340):(04390524484340)
    plt.figure()
#<1#(04390524484340)~%(1347837467) #>1#(03390595280198):(03390595280198)
    mnist_train_imagesCounts, mnist_train_imagesBins = np.histogram(mnist_train_images)
    plt.title('mnist_train_images Histogram')
    plt.xlabel('mnist_train_images')
    plt.ylabel('counts')
    plt.stairs(mnist_train_imagesCounts, mnist_train_imagesBins)
    # plt.show()
#<1#(03390595280198)~%(-2090729692) #>1#(03390524484340):(03390524484340)
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, columnspan=window.grid_size()[0], pady=10)
#<1#(03390524484340)~%(1036401314) #>1#(21390562142525):(21390562142525)
plot_button = Button(master = window,
    command = pixelHistogram,
    height = 2,
    width = 17,
    text = 'Pixel Histogram' )
plot_button.grid(row=0, column=1, padx=5)
window.columnconfigure( 1, weight=1)
#<1#(21390562142525)~%(-2030235503)


labels = [np.argmax(label) for label in mnist_training_labels]
#[Histogram labels]#@79049155
#[AddPlotToWindow]#@24795198
#[GUIButton]#@94119684
def labelHistogram():
#>1#(04390524795198):(04390524795198)
    plt.figure()
#<1#(04390524795198)~%(1347837467) #>1#(03390579049155):(03390579049155)
    labelsCounts, labelsBins = np.histogram(labels)
    plt.title('labels Histogram')
    plt.xlabel('labels')
    plt.ylabel('counts')
    plt.stairs(labelsCounts, labelsBins)
    # plt.show()
#<1#(03390579049155)~%(-1148697902) #>1#(03390524795198):(03390524795198)
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, columnspan=window.grid_size()[0], pady=10)
#<1#(03390524795198)~%(1036401314) #>1#(21390594119684):(21390594119684)
plot_button = Button(master = window,
    command = labelHistogram,
    height = 2,
    width = 17,
    text = 'Label Histogram' )
plot_button.grid(row=0, column=2, padx=5)
window.columnconfigure( 2, weight=1)
#<1#(21390594119684)~%(1577496143)


#[Visualize --loss]#@11487697
#[AddPlotToWindow]#@75403341
#[GUIButton]#@91524172
def plotLoss():
#>1#(04390575403341):(04390575403341)
    plt.figure()
#<1#(04390575403341)~%(1347837467) #>1#(03390511487697):(03390511487697)
    #***Plot history: Loss
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='val')
    plt.title('Train and Validation Loss History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.show()
#<1#(03390511487697)~%(-1958685116) #>1#(03390575403341):(03390575403341)
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, columnspan=window.grid_size()[0], pady=10)
#<1#(03390575403341)~%(1036401314) #>1#(21390591524172):(21390591524172)
plot_button = Button(master = window,
    command = plotLoss,
    height = 2,
    width = 11,
    text = 'Plot Loss' )
plot_button.grid(row=0, column=3, padx=5)
window.columnconfigure( 3, weight=1)
#<1#(21390591524172)~%(70811301)

#[Visualize --accuracy]#@28044886
#[AddPlotToWindow]#@81083293
#[GUIButton]#@35814401
def plotAccuracy():
#>1#(04390581083293):(04390581083293)
    plt.figure()
#<1#(04390581083293)~%(1347837467) #>1#(03390528044886):(03390528044886)
    #***Plot history: Accuracy
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='val')
    plt.title('Train and Validation Accuracy History')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.show()
#<1#(03390528044886)~%(-1536925120) #>1#(03390581083293):(03390581083293)
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, columnspan=window.grid_size()[0], pady=10)
#<1#(03390581083293)~%(1036401314) #>1#(21390535814401):(21390535814401)
plot_button = Button(master = window,
    command = plotAccuracy,
    height = 2,
    width = 15,
    text = 'Plot Accuracy' )
plot_button.grid(row=0, column=4, padx=5)
window.columnconfigure( 4, weight=1)
#<1#(21390535814401)~%(-36092845)

#[Visualize --confusionMatrix]#@63557867
#[AddPlotToWindow]#@49358333
#[GUIButton]#@88856461
def plotConfusionMatrix():
#>1#(04390549358333):(04390549358333)
    plt.figure()
#<1#(04390549358333)~%(1347837467) #>1#(03390563557867):(03390563557867)
    #***Visualize Confusion Matrix for Test Data
    confusionMatrixPrediction = model.predict(mnist_validation_images, batch_size=1)
    confusionMatrixActual = [np.argmax(label) for label in mnist_test_labels]
    confusionMatrixPrediction = [np.argmax(label) for label in confusionMatrixPrediction]
    
    confusionMatrixActual = [[label] for label in confusionMatrixActual]
    confusionMatrixPrediction = [[label] for label in confusionMatrixPrediction]
    
    skplt.metrics.plot_confusion_matrix(confusionMatrixActual, confusionMatrixPrediction, normalize=False, title = 'Confusion Matrix for mnist')
    skplt.metrics.plot_confusion_matrix(confusionMatrixActual, confusionMatrixPrediction, normalize=True, title = 'Normalized Confusion Matrix for mnist')
#<1#(03390563557867)~%(-1559729113) #>1#(03390549358333):(03390549358333)
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=0, columnspan=window.grid_size()[0], pady=10)
#<1#(03390549358333)~%(1036401314) #>1#(21390588856461):(21390588856461)
plot_button = Button(master = window,
    command = plotConfusionMatrix,
    height = 2,
    width = 23,
    text = 'Plot Confusion Matrix' )
plot_button.grid(row=0, column=5, padx=5)
window.columnconfigure( 5, weight=1)
#<1#(21390588856461)~%(-20336752) #>1#(23390568160469):(23390568160469)
window.mainloop()
#<1#(23390568160469)~%(-532138404)