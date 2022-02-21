from collections.abc import Iterable
from typing import Any, List, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    Layer,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from ..data import (
    ListOrInt,
    StringOrCallable,
    TensorOrNDArray,
    ThreeTensors,
    TupleOrInt,
)


def _get_conv_block(
    filters: int,
    kernel_size: TupleOrInt,
    strides: TupleOrInt,
    use_bias: bool,
    padding: str,
    initializer: StringOrCallable = "glorot_uniform",
) -> List[Layer]:
    """
    Returns a block consisting of A 2D Convolutional layer, a dropout layer and a batch normalization layer

    Args:
        filters:
        kernel_size:
        strides:
        use_bias:
        padding:
        initializer:

    Returns:

    """
    return [
        Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
            kernel_initializer=initializer,
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
    initializer: StringOrCallable = "glorot_uniform",
) -> List[Layer]:
    """
    Returns a block consisting of A 2D Convolutional Transpose layer, batch normalization layer

    Args:
        filters:
        kernel_size:
        strides:
        use_bias:
        padding:
        initializer:

    Returns:

    """

    return [
        Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
            kernel_initializer=initializer,
        ),
        LeakyReLU(),
    ]


class ConvEncoder:
    def __init__(self, layers: List[Layer]):
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

        return x


class ConvVariationalEncoder:
    def __init__(
        self,
        layers: List[Layer],
        name: str = "convolutional-encoder",
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


class ConvDecoder:
    def __init__(
        self,
        layers: List[Layer],
        name: str = "convolutional-decoder",
    ):
        self.layers = layers

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        return x


class ConvAutoEncoder(Model):
    name = "ConvAutoEncoder"

    def __init__(self, encoder: Model, decoder: Model, seed: int = None):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seed = seed

    # a hack for plotting a model
    @property
    def _model(self) -> Model:
        encoder_inputs = self.encoder.inputs
        encoder_outputs = self.encoder.outputs
        if isinstance(encoder_outputs, Iterable):
            z = encoder_outputs[0]
        else:
            z = encoder_outputs

        o = self.decoder(z)
        return Model(encoder_inputs, o)

    def call(self, inputs: TensorOrNDArray, training: bool = None):
        """Performs a single forward pass through the model.

        Args:
            inputs ([type]): [description]
            training ([type], optional): [description]. Defaults to None.
            mask ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        encoder_outputs = self.encoder(inputs, training=training)
        if isinstance(encoder_outputs, Iterable):
            z = encoder_outputs[0]
        else:
            z = encoder_outputs

        return self.decoder(z, training=training)

    def plot(self, fp: str) -> str:
        m = self._model()
        plot_model(m, to_file=fp, show_shapes=True, expand_nested=True)
        return fp

    def summary(self, line_length: int = None):
        return self._model.summary(line_length=line_length, expand_nested=True)

    def decode(self, inputs: TensorOrNDArray, return_array: bool = True) -> Any:
        o = self.decoder(inputs)
        if return_array:
            o = o.numpy()
        return o

    def encode(
        self, inputs: TensorOrNDArray, return_array: bool = True
    ) -> TensorOrNDArray:
        z = self.encoder(inputs)
        if return_array:
            z = z.numpy()
        return z

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            loss = self.loss(data, reconstruction)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "loss": loss,
        }

    @classmethod
    def for_128x128(
        cls, n_channels: int, latent_features: int = 1024, seed: int = None
    ) -> "ConvVAE":
        """Creates an instance of the Convolutional VAE for use with 128x128 images."""

        initializer = "glorot_uniform"
        if seed is not None:
            tf.random.set_seed(seed)
            initializer = GlorotUniform(seed=seed)

        a, b, c = 4, 4, 64
        reshape_into = (a, b, c)

        encoder_layers = [
            *_get_conv_block(
                32, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                64, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                128, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                256, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                512, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                1024, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            Flatten(),
            Dense(latent_features),
        ]

        decoder_layers = [
            Dense(a * b * c, activation="relu"),  # in (b, 1024)
            Reshape(reshape_into),  # (b, 4, 4, 64)
            *_get_conv_transpose_block(
                1024, 4, 2, False, "same", initializer=initializer
            ),  # (b, 8, 8, 1024)
            *_get_conv_transpose_block(
                512, 4, 2, False, "same", initializer=initializer
            ),  # (b, 16, 16, 512)
            *_get_conv_transpose_block(
                128, 4, 2, False, "same", initializer=initializer
            ),  # (b, 32, 32, 128)
            *_get_conv_transpose_block(
                64, 4, 2, False, "same", initializer=initializer
            ),  # (b, 64, 64, 128)
            Conv2DTranspose(
                n_channels,
                kernel_size=4,
                strides=2,
                use_bias=False,
                padding="same",
                activation=None,
                kernel_initializer=initializer,
            ),
        ]  # out (b, 128, 128, n_channels)

        encoder = ConvEncoder(layers=encoder_layers)
        decoder = ConvDecoder(layers=decoder_layers)

        # create the models
        inputs = Input((128, 128, n_channels))
        x = encoder(inputs)
        encoder_model = Model(inputs, x)

        rec = decoder(x)
        decoder_model = Model(x, rec)

        return cls(encoder_model, decoder_model)


class ConvVAE(ConvAutoEncoder):
    name = "ConvVAE"

    def encode(
        self, inputs: TensorOrNDArray, return_array: bool = True
    ) -> Tuple[Any, Any, Any]:
        z, means, logvars = self.encoder(inputs)
        if return_array:
            z = z.numpy()
            means = means.numpy()
            logvars = logvars.numpy()

        return z, means, logvars

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z, means, log_vars = self.encoder(data)
            reconstruction = self.decoder(z)
            loss = self.loss(data, reconstruction, means, log_vars)
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
    def for_128x128(
        cls, n_channels: int, latent_features: int = 1024, seed: int = None
    ) -> "ConvVAE":
        """Creates an instance of the Convolutional VAE for use with 128x128 images."""

        initializer = "glorot_uniform"
        if seed is not None:
            tf.random.set_seed(seed)
            initializer = GlorotUniform(seed=seed)

        a, b, c = 4, 4, 64
        reshape_into = (a, b, c)

        encoder_layers = [
            *_get_conv_block(
                32, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                64, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                128, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                256, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                512, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            *_get_conv_block(
                1024, 4, 2, use_bias=False, padding="same", initializer=initializer
            ),
            Flatten(),
            Dense(latent_features * 2),
        ]

        decoder_layers = [
            Dense(a * b * c, activation="relu"),  # in (b, 1024)
            Reshape(reshape_into),  # (b, 4, 4, 64)
            *_get_conv_transpose_block(
                1024, 4, 2, False, "same", initializer=initializer
            ),  # (b, 8, 8, 1024)
            *_get_conv_transpose_block(
                512, 4, 2, False, "same", initializer=initializer
            ),  # (b, 16, 16, 512)
            *_get_conv_transpose_block(
                128, 4, 2, False, "same", initializer=initializer
            ),  # (b, 32, 32, 128)
            *_get_conv_transpose_block(
                64, 4, 2, False, "same", initializer=initializer
            ),  # (b, 64, 64, 128)
            Conv2DTranspose(
                n_channels,
                kernel_size=4,
                strides=2,
                use_bias=False,
                padding="same",
                activation=None,
                kernel_initializer=initializer,
            ),
        ]  # out (b, 128, 128, n_channels)

        encoder = ConvVariationalEncoder(layers=encoder_layers)
        decoder = ConvDecoder(layers=decoder_layers)

        # create the models
        inputs = Input((128, 128, n_channels))
        x, means, logvars = encoder(inputs)
        encoder_model = Model(inputs, [x, means, logvars])

        rec = decoder(x)
        decoder_model = Model(x, rec)

        return cls(encoder_model, decoder_model)
