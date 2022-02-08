from typing import List, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.losses import Loss


class VAELoss(Loss):
    """ """

    def __init__(self, beta: float = 1.0):
        """

        :param beta: Controls the relative strength of the KL Divergence term in the total VAE loss.
                     If set to 1.0 (default) represents a regular VAE.
        """
        super(VAELoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE
        )  # otherwise doesn't return a Tuple
        self.beta = beta

    def kl_divergence(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """
        Computes the KL-divergence
        Args:
            mean:
            logvar:

        Returns:

        """
        return -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + logvar - tf.math.square(mean) - tf.math.exp(logvar), axis=1
            )
        )

    def call(
        self, y_true: tf.Tensor, outputs: List
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        def _as_tensors(*args):
            return [tf.convert_to_tensor(x) for x in args]

        reconstruction, mean, logvar = _as_tensors(*outputs)
        y_true, *_ = _as_tensors(y_true)

        kl_loss = tf.cast(self.kl_divergence(mean, logvar), tf.float32)

        bce_loss = tf.keras.backend.binary_crossentropy(y_true, reconstruction)
        bce_loss = tf.reduce_mean(bce_loss, axis=-1)
        bce_loss = tf.reduce_sum(bce_loss, axis=[1, 2])  # shape

        reconstruction_loss = tf.reduce_mean(bce_loss)

        return (
            reconstruction_loss + self.beta * kl_loss,
            reconstruction_loss,
            self.beta * kl_loss,
        )
