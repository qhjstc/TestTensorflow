"""
Fashion mnist test
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_image(ti, predictions_array, true_label, img):
    # plot images
    predictions_array, true_label, img = predictions_array, true_label[ti], img[ti]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[int(predicted_label)],
                                         100*np.max(predictions_array),
                                         class_names[true_label],
                                         color=color))


def plot_value_array(ti, predictions_array, true_label):
    # plot value array
    predictions_array, true_label = predictions_array, true_label[ti]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if __name__ == "__main__":
    # load fashion mnist
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # change data type
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # show top 25 images
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i])
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    # build tensorflow sequential
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)                       # output 10 data
    ])

    # set optimizer and loss function
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )

    # test and train
    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Add a softmax layer to convert Logits into more understandable probabilities
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()
                                             ])
    predictions = probability_model.predict(test_images)
    print("Test label is " + str(test_labels[0]))
    print("Prediction label is " + str(np.argmax(predictions[0])))

    # plot prediction
    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()


