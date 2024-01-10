import os
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import imageio

print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")

# Decide if to load an existing model or to train a new one
train_new_model = True

if train_new_model:
    # Loading the MNIST data set with samples and splitting it
    mnist = np.load('mnist.npz')
    (X_train, y_train), (X_test, y_test) = (mnist['x_train'], mnist['y_train']), (mnist['x_test'], mnist['y_test'])

    # Normalizing the data (making length = 1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    # Saving the model to a specific file
    model.save('handwritten_digits.h5')

else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.h5')

# Load custom images and predict them
image_number = 1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = imageio.imread(f'digits/digit{image_number}.png')[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("My OCR AI said it is {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1

# Correct indentation for the following line
print("Assumed numbers printed successfully.")
