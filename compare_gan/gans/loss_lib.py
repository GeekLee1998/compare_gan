# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of popular GAN losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compare_gan import utils
import numpy as np
import gin
import tensorflow as tf


def check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits):
  """Checks the shapes and ranks of logits and prediction tensors.

  Args:
    d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
    d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].

  Raises:
    ValueError: if the ranks or shapes are mismatched.
  """
  def _check_pair(a, b):
    if a != b:
      raise ValueError("Shape mismatch: %s vs %s." % (a, b))
    if len(a) != 2 or len(b) != 2:
      raise ValueError("Rank: expected 2, got %s and %s" % (len(a), len(b)))

  if (d_real is not None) and (d_fake is not None):
    _check_pair(d_real.shape.as_list(), d_fake.shape.as_list())
  if (d_real_logits is not None) and (d_fake_logits is not None):
    _check_pair(d_real_logits.shape.as_list(), d_fake_logits.shape.as_list())
  if (d_real is not None) and (d_real_logits is not None):
    _check_pair(d_real.shape.as_list(), d_real_logits.shape.as_list())


@gin.configurable(whitelist=[])
def non_saturating(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for Non-saturating loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("non_saturating_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real_logits, labels=tf.ones_like(d_real_logits),
        name="cross_entropy_d_real"))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits),
        name="cross_entropy_d_fake"))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.ones_like(d_fake_logits),
        name="cross_entropy_g"))
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def wasserstein(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for Wasserstein loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("wasserstein_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = -tf.reduce_mean(d_real_logits)
    d_loss_fake = tf.reduce_mean(d_fake_logits)
    d_loss = d_loss_real + d_loss_fake
    g_loss = -d_loss_fake
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def least_squares(d_real, d_fake, d_real_logits=None, d_fake_logits=None):
  """Returns the discriminator and generator loss for the least-squares loss.

  Args:
    d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
    d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
    d_real_logits: ignored.
    d_fake_logits: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("least_square_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.square(d_real - 1.0))
    d_loss_fake = tf.reduce_mean(tf.square(d_fake))
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    g_loss = 0.5 * tf.reduce_mean(tf.square(d_fake - 1.0))
    return d_loss, d_loss_real, d_loss_fake, g_loss


@gin.configurable(whitelist=[])
def hinge(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for the hinge loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("hinge_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
    d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
    d_loss = d_loss_real + d_loss_fake
    g_loss = - tf.reduce_mean(d_fake_logits)
    return d_loss, d_loss_real, d_loss_fake, g_loss

def lossfun(x, alpha, scale, approximate=False, epsilon=1e-6):
  """Implements the general form of the loss.
  This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.
  Args:
    x: The residual for which the loss is being computed. x can have any shape,
      and alpha and scale will be broadcasted to match x's shape if necessary.
      Must be a tensorflow tensor or numpy array of floats.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Must be a tensorflow tensor or
      numpy array of floats with the same precision as `x`. Varying alpha allows
      for smooth interpolation between a number of discrete robust losses:
      alpha=-Infinity: Welsch/Leclerc Loss.
      alpha=-2: Geman-McClure loss.
      alpha=0: Cauchy/Lortentzian loss.
      alpha=1: Charbonnier/pseudo-Huber loss.
      alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha. Must be a tensorflow tensor or numpy
      array of single-precision floats.
    approximate: a bool, where if True, this function returns an approximate and
      faster form of the loss, as described in the appendix of the paper. This
      approximation holds well everywhere except as x and alpha approach zero.
    epsilon: A float that determines how inaccurate the "approximate" version of
      the loss will be. Larger values are less accurate but more numerically
      stable. Must be great than single-precision machine epsilon.
  Returns:
    The losses for each element of x, in the same shape as x. This is returned
    as a TensorFlow graph node of single precision floats.
  """
  # `scale` and `alpha` must have the same type as `x`.
  float_dtype = x.dtype
  tf.debugging.assert_type(scale, float_dtype)
  tf.debugging.assert_type(alpha, float_dtype)
  # `scale` must be > 0.
  assert_ops = [tf.Assert(tf.reduce_all(tf.greater(scale, 0.)), [scale])]
  with tf.control_dependencies(assert_ops):
    # Broadcast `alpha` and `scale` to have the same shape as `x`.
    alpha = tf.broadcast_to(alpha, tf.shape(x))
    scale = tf.broadcast_to(scale, tf.shape(x))

    if approximate:
      # `epsilon` must be greater than single-precision machine epsilon.
      assert epsilon > np.finfo(np.float32).eps
      # Compute an approximate form of the loss which is faster, but innacurate
      # when x and alpha are near zero.
      b = tf.abs(alpha - tf.cast(2., float_dtype)) + epsilon
      d = tf.where(
          tf.greater_equal(alpha, 0.), alpha + epsilon, alpha - epsilon)
      loss = (b / d) * (tf.pow(tf.square(x / scale) / b + 1., 0.5 * d) - 1.)
    else:
      # Compute the exact loss.

      # This will be used repeatedly.
      squared_scaled_x = tf.square(x / scale)

      # The loss when alpha == 2.
      loss_two = 0.5 * squared_scaled_x
      # The loss when alpha == 0.
      loss_zero = utils.log1p_safe(0.5 * squared_scaled_x)
      # The loss when alpha == -infinity.
      loss_neginf = -tf.math.expm1(-0.5 * squared_scaled_x)
      # The loss when alpha == +infinity.
      loss_posinf = utils.expm1_safe(0.5 * squared_scaled_x)

      # The loss when not in one of the above special cases.
      machine_epsilon = tf.cast(np.finfo(np.float32).eps, float_dtype)
      # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
      beta_safe = tf.maximum(machine_epsilon, tf.abs(alpha - 2.))
      # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
      alpha_safe = tf.where(
          tf.greater_equal(alpha, 0.), tf.ones_like(alpha),
          -tf.ones_like(alpha)) * tf.maximum(machine_epsilon, tf.abs(alpha))
      loss_otherwise = (beta_safe / alpha_safe) * (
          tf.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

      # Select which of the cases of the loss to return.
      loss = tf.where(
          tf.equal(alpha, -tf.cast(float('inf'), float_dtype)), loss_neginf,
          tf.where(
              tf.equal(alpha, 0.), loss_zero,
              tf.where(
                  tf.equal(alpha, 2.), loss_two,
                  tf.where(
                      tf.equal(alpha, tf.cast(float('inf'), float_dtype)),
                      loss_posinf, loss_otherwise))))

    return loss

@gin.configurable(whitelist=[])
def robust_loss(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  with tf.name_scope("robust_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    alpha = np.float64(
      np.round(np.random.uniform(-16, 3) * 10) / 10.)
    scale = np.float64(
      np.exp(np.random.normal(size=None) * 4.) + 1e-5)
    d_loss_real = lossfun(d_real_logits,alpha,scale)
    d_loss_fake = lossfun(d_fake_logits,alpha,scale)
    d_loss = d_loss_real + d_loss_fake
    g_loss = lossfun(d_fake_logits,alpha,scale)
    return d_loss, d_loss_real, d_loss_fake, g_loss

@gin.configurable("loss", whitelist=["fn"])
def get_losses(fn=non_saturating, **kwargs):
  """Returns the losses for the discriminator and generator."""
  return utils.call_with_accepted_args(fn, **kwargs)
