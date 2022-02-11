from abc import ABC, abstractclassmethod
from typing import Callable

import tensorflow as tf
from tensorflow import Tensor

from ..data import ThreeTensors

tmean = tf.math.reduce_mean
sq = tf.math.square
exp = tf.math.exp
tsum = tf.math.reduce_sum
log = tf.math.log
pow = tf.math.pow

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.mse


class _VAELossBase(ABC):
    def __init__(
        self, beta: float = 1.0, scaling: float = 1e-4, kl_reduction: Callable = tsum
    ) -> None:
        """Base class for implmenting variations of the VAE losses"""

        self.scaling = scaling
        self.beta = beta
        self.kl_reduction = kl_reduction

    def kld_w_std_normal(self, means: Tensor, logvars: Tensor) -> Tensor:
        """Computes the KL Divergence between the distribution N(means, I*logvars) and the standard normal distribution of the same dimension.

        Args:
            means (Tensor): A vector of means characterizing the distribution of the latent space.
            logvars (Tensor): A vector of the logarithms of variences characterizing the distribution of the latent space.

        Returns:
            Tensor: A tensor that carries the kl divergence between the target and input distributions
        """

        return -0.5 * self.kl_reduction(1 + logvars - sq(means) - exp(logvars))

    @abstractclassmethod
    def reconstruction_loss(self, image: Tensor, reconstruction: Tensor) -> Tensor:
        """Returns the loss associated with the reconstruction error between the original and reconstructed images.

        Args:
            image (Tensor): The original images.
            reconstruction (Tensor): The reconstructions of the original images.

        Returns:
            Tensor: A tensor contatining the reconstruction loss
        """

        raise NotImplementedError

    def __call__(
        self, image: Tensor, reconstruction: Tensor, means: Tensor, logvars: Tensor
    ) -> ThreeTensors:
        kld_loss_term = self.scaling * self.beta * self.kld_w_std_normal(means, logvars)
        reconstruction_loss_term = self.scaling * self.reconstruction_loss(
            image, reconstruction
        )

        return (
            reconstruction_loss_term + kld_loss_term,
            reconstruction_loss_term,
            kld_loss_term,
        )

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        config = dict(
            scaling=self.scaling, beta=self.beta, kl_reduction=self.kl_reduction
        )
        for k, v in config.items():
            s += f"{k}={v}, "

        s = s[:-2] + ")"
        return s

    def __str__(self):
        return self.__repr__()


class VAELoss(_VAELossBase):
    def __init__(self, beta: float = 1, scaling: float = 0.0001) -> None:
        """Implements VAE Loss. Optional beta parameter can be used to enforce greater or smaller penalty on the KL Divergence term of the total loss.

        Args:
            beta (float, optional):  Controls the impact of the KL Divergence on the total VAE loss. Defaults to 1.
            scaling (float, optional): A scalar value that will be applied to the total loss (reconstruction + kld). Defaults to 1e-4.
            kl_reduction (Callable, optional): Operation to perform on the vector of KL divergence values. Defatuls to tmean
        """
        super().__init__(beta, scaling, kl_reduction=tmean)

    def reconstruction_loss(self, image: Tensor, reconstruction: Tensor) -> Tensor:
        """Implements the optimal sigma loss function from https://arxiv.org/abs/2006.13202

        Args:
            image (Tensor): Original images
            reconstruction (Tensor): Reconstructed images

        Returns:
            Tensor: Tensor containing reconstruction loss
        """
        mse_loss = mse(image, reconstruction)
        return tsum(mse_loss)


class SigmaVAELoss(_VAELossBase):
    def __init__(self, beta: float = 1, scaling: float = 0.0001) -> None:
        """Initializes the Sigma VAE loss function. Optional beta parameter can be use to further add or lower penalty to the KL Divergence term.

        Args:
            beta (float, optional): Controls the impact of the KL Divergence on the total VAE loss. Defaults to 1.
            scaling (float, optional): Scaling applied to the total loss. Defaults to 0.0001.
        """
        super(SigmaVAELoss, self).__init__(beta, scaling, kl_reduction=tmean)

    def reconstruction_loss(self, image: Tensor, reconstruction: Tensor) -> Tensor:
        """Implements the optimal sigma loss function from https://arxiv.org/abs/2006.13202

        Args:
            image (Tensor): Original images
            reconstruction (Tensor): Reconstructed images

        Returns:
            Tensor: Tensor containing reconstruction loss
        """

        mse_loss = tmean(mse(image, reconstruction))
        log_sigma_opt = 0.5 * log(mse_loss)
        r_loss = (
            0.5 * pow((image - reconstruction) / exp(log_sigma_opt), 2) + log_sigma_opt
        )
        return tsum(r_loss)
