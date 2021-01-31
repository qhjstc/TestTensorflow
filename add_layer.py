"""
Build the first network
it comes from https://www.bilibili.com/video/BV1Lx411j7ws?p=16
"""
import tensorflow as tsf
import numpy as np
import matplotlib.pyplot as plt

tsf.compat.v1.disable_eager_execution()       # in tensorflow 2.0 eager execution is enabled by default
tf = tsf.compat.v1            # change the version


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]))       # Create a random matrix
            tf.summary.histogram(layer_name+':Weight', weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)        # Generally, it is not set to 0, so we set it to 0.1
        with tf.name_scope('Wx_plus_b'):
            wx_plus_b = tf.matmul(inputs, weights) + biases           # matrix multiplication, but i have some questions in here
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)             # Excitation equation
        return outputs


if __name__ == "__main__":
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]         # input only has one element , so it only has one neuron
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # define placeholder for input to network
    with tf.name_scope('input'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    # add hidden layer
    l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.tanh)
    # add output layer
    prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),  # Calculate the average sum
                                            reduction_indices=[1]))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)     # reduce loss
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        # And then use terminal to input "tensorboard --logdir=logs"
        # Pay attention to run in the root directory, not in logs
        sess.run(init)

        fig = plt.figure()  # build a figure
        ax = fig.add_subplot(1, 1, 1)  # Add a subplot
        ax.scatter(x_data, y_data)
        plt.ion()  # Make the display continuous
        # plt.show(block=False)
        for i in range(1000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 50 == 0:
                # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={xs: x_data})
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)            # lw is width
                plt.pause(0.1)

                # add result
                result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
                writer.add_summary(result, i)
        plt.ioff()  # Close interactive mode
        # plt.pause(0)


