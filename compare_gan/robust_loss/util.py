import numpy as np
import tensorflow as tf



def log1p_safe(x):
  """The same as tf.math.log1p(x), but clamps the input to prevent NaNs."""
  return tf.math.log1p(tf.minimum(x, tf.cast(3e37, x.dtype)))


def exp_safe(x):
  """The same as tf.math.exp(x), but clamps the input to prevent NaNs."""
  return tf.math.exp(tf.minimum(x, tf.cast(87.5, x.dtype)))