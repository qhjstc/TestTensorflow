"""
Build the first network
it comes from https://www.bilibili.com/video/BV1Lx411j7ws?p=16
"""
import tensorflow as tsf
import numpy as np

tsf.compat.v1.disable_eager_execution()       # in tensorflow 2.0 eager execution is enabled by default
tf = tsf.compat.v1            # change the version


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))       # Create a random matrix
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)        # Generally, it is not set to 0, so we set it to 0.1
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

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)        # build 10 neuron in the first layer
    prediction = add_layer(l1, 10, 1, activation_function=None)      # build the second layer

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)     # reduce loss
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 50 == 0:
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
