#
# Main Tensorflow Code Loop
#

########
########  Setup Environment and install libs
########


####
#### Installing Required Libraries
####
!pip install tensorflow --progress-bar off
!pip install scikit-learn pandas --progress-bar off

####
#### The follow line is required to fix an issue with the latest Cuda drivers and tensorflow.
####
!ln -s /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11.0.1.105 /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.10


# import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

# Check if CUDA is loaded properly
tf.config.list_physical_devices('GPU')


########
########  Build and train model
########


####
#### Setup the Dataset
####


# leverage the dataset from keras module
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# convert to floats first
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Normalise image to range 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

####
#### Setup the Model and training task
####

# Instantiate the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-05),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(600),
    tf.keras.layers.Dropout(rate=0.25),
    tf.keras.layers.Dense(120),
    tf.keras.layers.Dense(10)
])

# setup loss and optimisers
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

####
#### Train the Model
####

###### Train the Model
history = model.fit(train_images, train_labels, epochs=5)

###### Evaluate the Model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

####
#### Look at model train result
####

#### Visualise the training Loss
plt.plot(history.history['loss'])

#### Visualise the training accuracy
plt.plot(history.history['accuracy'])

### Confusion Matrix Report
prediction_raw = model.predict(test_images)
prediction_labels = tf.argmax(prediction_raw, axis=1)

confusion_matrix(test_labels, prediction_labels)
print("Classification report for CNN :\n%s\n"
      % (metrics.classification_report(test_labels, prediction_labels)))
