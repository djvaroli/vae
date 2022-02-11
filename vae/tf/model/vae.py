from typing import List, Tuple, Any

import tensorflow as tf
from tensorflow import Tensor
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

from tf.data import (
    ListOrInt,
    TensorOrNDArray,
    ThreeTensors,
    TupleOrInt,
)


def _get_conv_block(
    filters: int,
    kernel_size: TupleOrInt,
    strides: TupleOrInt,
    use_bias: bool,
    padding: str
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
        LeakyReLU()
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
            padding=padding
        ),
        LeakyReLU()
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

    def __init__(
        self,
        encoder: Model,
        decoder: Model
    ):
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

        z, mean, logvars = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def encode(self, inputs: TensorOrNDArray, return_array: bool = True) -> Tuple[Any, Any, Any]:
        z, means, logvars = self.encoder(inputs)
        if return_array:
          z = z.numpy()
          means = means.numpy()
          logvars = logvars.numpy()
        o = (z, means, logvars)

        return o

    def decode(self, inputs: TensorOrNDArray, return_array: bool = True) -> Any:
        o = self.decoder(inputs)
        if return_array:
          o = o.numpy()
        return o

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

    def plot(self, fp: str) -> str:
      m = self._model()
      plot_model(m, to_file=fp, show_shapes=True, expand_nested=True)
      return fp
    
    @classmethod
    def for_128x128(cls, n_channels: int, latent_features: int = 1024) -> "CVAE":
        """Creates an instance of the Convolutional VAE for use with 128x128 images.
        """
        
        a, b, c = 4, 4, 64
        reshape_into = (a, b, c)

        encoder_layers = [
            *_get_conv_block(32, 4, 2, use_bias=False, padding="same"),
            *_get_conv_block(64, 4, 2, use_bias=False, padding="same"),
            *_get_conv_block(128, 4, 2, use_bias=False, padding="same"),
            *_get_conv_block(256, 4, 2, use_bias=False, padding="same"),
            *_get_conv_block(512, 4, 2, use_bias=False, padding="same"),
            *_get_conv_block(1024, 4, 2, use_bias=False, padding="same"),
            Flatten(),
            Dense(latent_features * 2),
        ]
        
        decoder_layers = [
            Dense(a * b * c, activation="relu"),  # in (b, 1024)
            Reshape(reshape_into),  # (b, 4, 4, 64)
            *_get_conv_transpose_block(1024, 4, 2, False, "same"),  # (b, 8, 8, 1024)
            *_get_conv_transpose_block(512, 4, 2, False, "same"),  # (b, 16, 16, 512)
            *_get_conv_transpose_block(128, 4, 2, False, "same"),  # (b, 32, 32, 128)
            *_get_conv_transpose_block(64, 4, 2, False, "same"),  # (b, 64, 64, 128)
            Conv2DTranspose(
                n_channels, kernel_size=4, strides=2, use_bias=False, padding="same", activation=None,
            ),
        ]  # out (b, 128, 128, n_channels)
        
        encoder = CEncoder(layers=encoder_layers)
        decoder = CDecoder(layers=decoder_layers)
        
        # create the models
        inputs = Input((128, 128, n_channels))
        x, means, logvars = encoder(inputs)
        encoder_model = Model(inputs, [x, means, logvars])
        
        rec = decoder(x)
        decoder_model = Model(x, rec)
        
        return cls(encoder_model, decoder_model)
