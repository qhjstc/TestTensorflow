"""
Mnist data set train
code from https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh-cn
"""

import tensorflow as tf


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = x_train/255.0, y_train/255.0

    model = tf.keras.Sequential([                          # build tensorflow sequential
        tf.keras.layers.Flatten(input_shape=(28, 28)),     # flatten data
        tf.keras.layers.Dense(128, activation='relu'),     # like v1 Weight and biases
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')    # set activation function
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',         # add optimizer adn loss function
                  metrics=['accuracy'])

    # test and train
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)






