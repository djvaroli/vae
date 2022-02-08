from typing import List, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten, Input,
                                     Layer, LeakyReLU, Reshape)
from tensorflow.keras.models import Model

from vae.tf.utilites import make_list_if_not
from vae.tf.utilites.data_types import (ListOrInt, StringOrCallable,
                                        TensorOrNDArray, ThreeTensors,
                                        TupleOrInt)


# @title Model Blocks
def _get_conv_block(
    filters: int,
    kernel_size: TupleOrInt,
    strides: TupleOrInt,
    use_bias: bool,
    padding: str,
) -> List[Layer]:
    """
    Returns a block consisting of A 2D Convolutional layer, a dropout layer and a batch normalization layer

    Args:
        filters:
        kernel_size:
        strides:
        use_bias:
        padding:

    Returns:

    """
    return [
        Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
        ),
        BatchNormalization(),
        LeakyReLU(),
    ]


def _get_conv_transpose_block(
    filters: int,
    kernel_size: ListOrInt,
    strides: ListOrInt,
    use_bias: bool,
    padding: str,
) -> List[Layer]:
    """
    Returns a block consisting of A 2D Convolutional Transpose layer, batch normalization layer

    Args:
        filters:
        kernel_size:
        strides:
        use_bias:
        padding:
        dropout_rate:

    Returns:

    """

    return [
        Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
        ),
        LeakyReLU(),
    ]


class CEncoder:
    def __init__(
        self, layers: List[Layer], name: str = "convolutional-encoder",
    ):
        self.layers = layers

    def __call__(self, inputs: Tensor, *args, **kwargs) -> ThreeTensors:
        """
        Performs the forward pass on the Encoder.
        Args:
            inputs: Tensor to be as used as inputs to the encoder
            *args:
            **kwargs:

        Returns:
            Tensor of latent embedded representation of input data
            Tensor of means
            Tensor of log variances
        """
        x = inputs
        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        return self.reparameterize(x)

    def reparameterize(self, means_log_vars: Tensor) -> ThreeTensors:
        """
        Performs the reparameterization step given a Tensor of means and log variances
        Args:
            means_log_vars: Tensor of means and log variances along the axis 1

        Returns:
            Tensor of latent embedded representation of input data
            Tensor of means
            Tensor of log variances
        """
        means, log_vars = tf.split(means_log_vars, num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=tf.shape(means))
        z = means + tf.exp(log_vars * 0.5) * eps
        return z, means, log_vars


class CDecoder:
    def __init__(
        self, layers: List[Layer], name: str = "convolutional-decoder",
    ):
        self.layers = layers

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        return x


class CVAE(Model):
    name = "CVAE"

    def __init__(self, encoder: Model, decoder: Model):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # a hack for plotting a model
    def _model(self):
        i = self.encoder.inputs
        z, means, log_vars = self.encoder.outputs
        o = self.decoder(z)
        return Model(i, o)

    def call(self, inputs: TensorOrNDArray, training: bool = None, mask=None):
        """Performs a single forward pass through the model.

        Args:
            inputs ([type]): [description]
            training ([type], optional): [description]. Defaults to None.
            mask ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        z, mean, logvars = self.encode(inputs, training=training)
        return self.decode(z, training=training)

    def encode(self, inputs: TensorOrNDArray, training: bool = None) -> ThreeTensors:
        return self.encoder(inputs, training=training)

    def decode(self, inputs: TensorOrNDArray, training: bool = None) -> Tensor:
        return self.decoder(inputs, training=training)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z, means, log_vars = self.encoder(data)
            reconstruction = self.decoder(z)
            loss = self.loss(data, [reconstruction, means, log_vars])
            total_loss, reconstruction_loss, kl_loss = loss[0], loss[1], loss[2]

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    @classmethod
    def for_MNIST(
        cls,
        latent_features: int = 2,
        dropout_rate: float = 0.0,
        output_activation: StringOrCallable = "sigmoid",
    ) -> "CVAE":
        """Returns an instance of a Conv VAE for use on the MNIST dataset.
        Assumes images are of shape (28, 28, 1)
        """
        a, b, c = 7, 7, 64
        reshape_into = (a, b, c)

        encoder_layers = [
            *_get_conv_block(256, 3, 1, use_bias=False, padding="same"),
            *_get_conv_block(128, 3, 1, use_bias=False, padding="same"),
            *_get_conv_block(64, 3, 2, use_bias=False, padding="same"),
            *_get_conv_block(32, 3, 2, use_bias=False, padding="same"),
            Flatten(),
            Dense(latent_features * 2),
        ]

        decoder_layers = [
            Dense(a * b * c),
            Reshape(reshape_into),
            *_get_conv_transpose_block(64, 5, 1, False, "same", dropout_rate),
            *_get_conv_transpose_block(128, 5, 2, False, "same", dropout_rate),
            *_get_conv_transpose_block(256, 3, 1, False, "same", dropout_rate),
            *_get_conv_transpose_block(512, 3, 1, False, "same", dropout_rate),
            Conv2DTranspose(
                1,
                kernel_size=3,
                strides=2,
                use_bias=False,
                padding="same",
                activation=output_activation,
            ),
        ]

        encoder = CEncoder(layers=encoder_layers)
        decoder = CDecoder(layers=decoder_layers)

        # create the models
        inputs = Input((28, 28, 1))
        x, means, logvars = encoder(inputs)
        encoder_model = Model(inputs, [x, means, logvars])

        rec = decoder(x)
        decoder_model = Model(x, rec)

        return cls(encoder_model, decoder_model)

    @classmethod
    def for_SIMPSONS(
        cls,
        latent_features: int = 50,
        dropout_rate: float = 0.0,
        output_activation: StringOrCallable = "sigmoid",
    ) -> "CVAE":
        """Creates an instance of a CVAE for use with the Simpsons Faces dataset
        """
        # target shape 200 200 3

        a, b, c = 25, 25, 4
        reshape_into = (a, b, c)

        encoder_layers = [
            *_get_conv_block(8, 5, 1, use_bias=False, padding="same"),
            *_get_conv_block(16, 5, 1, use_bias=False, padding="same"),
            *_get_conv_block(32, 5, 1, use_bias=False, padding="same"),
            *_get_conv_block(64, 5, 2, use_bias=False, padding="same"),
            *_get_conv_block(128, 5, 2, use_bias=False, padding="same"),
            *_get_conv_block(256, 5, 2, use_bias=False, padding="same"),
            *_get_conv_block(512, 5, 2, use_bias=False, padding="same"),
            Flatten(),
            Dense(latent_features * 2),
        ]

        decoder_layers = [
            Dense(a * b * c),
            Reshape(reshape_into),
            *_get_conv_transpose_block(512, 5, 1, False, "same", dropout_rate),
            *_get_conv_transpose_block(256, 5, 1, False, "same", dropout_rate),
            *_get_conv_transpose_block(128, 5, 1, False, "same", dropout_rate),
            *_get_conv_transpose_block(128, 5, 1, False, "same", dropout_rate),
            *_get_conv_transpose_block(128, 5, 2, False, "same", dropout_rate),
            *_get_conv_transpose_block(64, 5, 2, False, "same", dropout_rate),
            Conv2DTranspose(
                3,
                kernel_size=3,
                strides=2,
                use_bias=False,
                padding="same",
                activation=output_activation,
            ),
        ]

        encoder = CEncoder(layers=encoder_layers)
        decoder = CDecoder(layers=decoder_layers)

        # create the models
        inputs = Input((200, 200, 3))
        x, means, logvars = encoder(inputs)
        encoder_model = Model(inputs, [x, means, logvars])

        rec = decoder(x)
        decoder_model = Model(x, rec)

        return cls(encoder_model, decoder_model)

    @classmethod
    def for_ANIME(
        cls, latent_features: int = 50, output_activation: StringOrCallable = None
    ) -> "CVAE":
        """[summary]
        """

        a, b, c = 8, 8, 64
        reshape_into = (a, b, c)

        encoder_layers = [
            *_get_conv_block(32, 5, 1, use_bias=False, padding="same"),
            *_get_conv_block(64, 5, 1, use_bias=False, padding="same"),
            *_get_conv_block(64, 5, 2, use_bias=False, padding="same"),
            *_get_conv_block(64, 5, 2, use_bias=False, padding="same"),
            Flatten(),
            Dense(latent_features * 2),
        ]

        decoder_layers = [
            Dense(a * b * c),
            Reshape(reshape_into),
            *_get_conv_transpose_block(256, 5, 2, False, "same"),
            *_get_conv_transpose_block(128, 5, 2, False, "same"),
            *_get_conv_transpose_block(64, 5, 2, False, "same"),
            *_get_conv_transpose_block(32, 5, 1, False, "same"),
            Conv2DTranspose(
                3,
                kernel_size=5,
                strides=2,
                use_bias=False,
                padding="same",
                activation=output_activation,
            ),
        ]

        encoder = CEncoder(layers=encoder_layers)
        decoder = CDecoder(layers=decoder_layers)

        # create the models
        inputs = Input((28, 28, 1))
        x, means, logvars = encoder(inputs)
        encoder_model = Model(inputs, [x, means, logvars])

        rec = decoder(x)
        decoder_model = Model(x, rec)

        return cls(encoder_model, decoder_model)
