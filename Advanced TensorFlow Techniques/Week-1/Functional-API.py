import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt

import pydot
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model

def build_model_with_sequential():

    # instantiate a Sequential class and linearly stack the layers of your model
    seq_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )
    return seq_model


def build_model_with_functional():

    # instantiate the input Tensor
    input_layer = tf.keras.Input(shape=(28, 28))

    # stack the layers using the syntax: new_layer()(previous_layer)
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    first_dense = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
    output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(first_dense)

    # declare inputs and outputs
    func_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return func_model

model = build_model_with_functional()
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')


# prepare fashion mnist dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0
def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()
show_image(training_images[100])
# configure, train, and evaluate the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
