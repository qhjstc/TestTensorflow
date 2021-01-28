
import tensorflow as tsf

tf = tsf.compat.v1            # change the version


def add_layer(inputs, in_size, out_size, activation_function = None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))       # Create a random matrix
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)        # Generally, it is not set to 0, so we set it to 0.1
    wx_plus_b = tf.matmul(inputs, weights) + biases           # matrix multiplication, but i have some questions in here
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)             # Excitation equation
    return outputs

