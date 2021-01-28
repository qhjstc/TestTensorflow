"""
This is the fist time to use Tensorflow

This example is from BilBil https://www.bilibili.com/video/BV1Lx411j7ws?p=9


"""
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()       # in tensorflow 2.0 eager execution is enabled by default

# creat data
x_data = np.random.rand(100).astype(np.float32)     # Tensorflow's general data format is float32
y_data = x_data*0.1 + 0.3


# create tensorflow structure start #
Weight = tf.compat.v1.Variable(tf.compat.v1.random_uniform([1], -1.0, 1.0))  # Build a data 1 to minus -1
biases = tf.compat.v1.Variable(tf.compat.v1.zeros([1]))

y = Weight * x_data + biases

loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(y-y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
# Because of the version problem. There is a big difference between 1 and 2
train = optimizer.minimize(loss)                                      # build a optimizer

init = tf.compat.v1.global_variables_initializer()
# create tensorflow structure end #

sess = tf.compat.v1.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weight), sess.run(biases))

sess.close()
