# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:

    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.prediction
        self.optimize
        self.error

    # @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    # def prediction(self):
    #     x = self.text
    #     x = tf.contrib.slim.fully_connected(x, 200)
    #     x = tf.contrib.slim.fully_connected(x, 200)
    #     x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
    #     return x

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        n_steps = int(self.text.get_shape()[1])
        n_classes = int(self.label.get_shape()[1])

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(self.text, n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Define weight and bias

        weight = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[n_classes]))

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weight) + bias

    @define_scope
    def optimize(self):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction,
                                                                      labels=self.label))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        return optimizer


    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
# n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
# n_classes = 10 # MNIST total classes (0-9 digits)


def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    text = tf.placeholder(tf.float32, [None, 28, 28])
    label = tf.placeholder(tf.float32, [None, 10])
    model = Model(text, label)

    # config = SmallConfig()


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(10):
      texts, labels = mnist.test.images, mnist.test.labels
      texts = texts.reshape((batch_size, 28, 28))
      error = sess.run(model.error, {text: texts, label: labels})
      print('Test error {:6.2f}%'.format(100 * error))
      for _ in range(10000):
        texts, labels = mnist.train.next_batch(100)
        texts = texts.reshape((batch_size, 28, 28))
        sess.run(model.optimize, {text: texts, label: labels})



if __name__ == '__main__':
  main()