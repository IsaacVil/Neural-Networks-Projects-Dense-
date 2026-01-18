import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import sklearn


def normX(x):
    m = x.shape[0]
    unos = np.ones(m)
    miu = (x.T @ unos) / m
    sigma = np.sqrt((unos @ ((x - miu)**2)) / m) #multiplicacion de matriz (solo permite n1xm1 * n2xm2 en el que m1 = n2)
    sigma[sigma == 0] = 1
    z = (x - miu) / sigma
    return z, miu, sigma

#Tensorflow solo funciona con python 3.11 hacia abajo, la que uso es 3.11.9, py -3.11.9 NumberRecogNeuralNetwork.py

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#reusar normX
X_train = X_train.reshape(len(X_train), -1)
X, miuX, sigmaX = normX(X_train[:60000, :])
y = y_train[:60000]

model = tensorflow.keras.Sequential(
    [
        tf.keras.layers.Dense(units=100, activation="relu"),
        tf.keras.layers.Dense(units=70, activation="relu"),
        tf.keras.layers.Dense(units=50, activation="relu"),
        tf.keras.layers.Dense(units=30, activation="relu"),
        tf.keras.layers.Dense(units=15, activation="relu"),
        tf.keras.layers.Dense(units=10, activation="linear"),
    ]
    )

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True))
model.fit(X, y, epochs=60)


testNormX = (X_test.reshape(len(X_test), -1) - miuX)/sigmaX



predictions = model.predict(testNormX) #la vuelve de (, 28, 28) a (, 28*28 )
predictionsArgmax = np.zeros(predictions.shape[0], dtype=int)
for i in range(predictions.shape[0]):
    predictionsArgmax[i] = np.argmax(predictions[i])
accuracy = np.mean(predictionsArgmax == y_test)
print(predictionsArgmax[10], y_test[10])
print("Accuracy:", accuracy)

model.save(".\\data\\28x28paint.keras")
np.save(".\\data\\miuX.npy", miuX)
np.save(".\\data\\sigmaX.npy", sigmaX)