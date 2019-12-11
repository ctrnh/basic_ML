import tensorflow as tf
import functools

## Doublewrap and define_scope function from https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
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


class Model3:
    def __init__(self, Xtf, ytf,learning_rate,param_layers):
        self.Xtf = Xtf
        self.ytf = ytf
        self.learning_rate = learning_rate
        self.param_layers = param_layers
        self.prediction 
        self.optimize 
        self.accuracy 
        self.loss 

        
    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        filters1, kernel_size1, padding1,filters2, kernel_size2, padding2, mlp_units1,mlp_units2 = self.param_layers.values()
        
        conv1 = tf.layers.conv2d(
                inputs=self.Xtf,
                filters=filters1,
                kernel_size=[kernel_size1, kernel_size1],
                padding=padding1,
                activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=filters2,
                kernel_size=[kernel_size2, kernel_size2],
                padding=padding2,
                activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.layers.Flatten()(pool2)

        mlp = tf.layers.dense(inputs=pool2_flat, units=mlp_units1, activation=tf.nn.relu)

        mlp2 = tf.layers.dense(inputs=mlp, units=mlp_units2, activation=tf.nn.relu)

        logits = tf.layers.dense(inputs=mlp2, units=10)
        self._prediction = tf.nn.softmax(logits)
        return (self._prediction)

    @define_scope
    def optimize(self):
        if 0==0:
            learning_rate = self.learning_rate
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.ytf, logits=tf.log(self.prediction))
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self._optimize = optimizer.minimize(loss)
        return self._optimize

    @define_scope
    def loss(self):
        if 0==0:
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.ytf, logits=tf.log(self.prediction))
            self._loss = loss
        return self._loss

    @define_scope
    def accuracy(self):
        if 0==0:
            equal = tf.equal(self.ytf, tf.argmax(self.prediction, 1))
            self._accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
        return self._accuracy

