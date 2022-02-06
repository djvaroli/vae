from typing import List

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model

from vae.tf.utilites import make_list_if_not
from vae.tf.utilites.data_types import TupleOrInt, ThreeTensors, ListOrInt, StringOrCallable, Tuple, TensorOrNDArray

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


def _get_conv_transpose_blocks(
    filters: List,
    kernel_size: ListOrInt,
    strides: ListOrInt,
    use_bias: bool,
    padding: str,
    dropout_rate: float,
    output_activation: str,
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
        output_activation

    Returns:

    """
    layers = []
    for filters_ in filters[:-1]:
        layers.extend(
            [
                Conv2DTranspose(
                    filters_,
                    kernel_size=kernel_size,
                    strides=strides,
                    use_bias=use_bias,
                    padding=padding,
                ),
                LeakyReLU(),
                Dropout(dropout_rate),
            ]
        )

    layers.append(
        Conv2DTranspose(
            filters[-1],
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
            activation=output_activation,
        )
    )

    return layers


class CEncoder:
    def __init__(
        self,
        latent_features: int,
        filters: ListOrInt,
        kernel_size: TupleOrInt = (3, 3),
        strides: TupleOrInt = (2, 2),
        use_bias: bool = False,
        padding: str = "same",
        dropout_rate: float = 0.0,
        name: str = "convolutional-encoder",
    ):
        self.name = name
        self.latent_features = latent_features
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.filters = make_list_if_not(filters)

        self.layers = []
        for filters_ in self.filters:
            self.layers.extend(
                _get_conv_block(filters_, kernel_size, strides, use_bias, padding)
            )

        self.layers.extend([Flatten(), Dense(self.latent_features * 2)])

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
        self,
        reshape_into: Tuple[int, int, int],
        filters: ListOrInt,
        kernel_size: TupleOrInt = (3, 3),
        strides: TupleOrInt = (2, 2),
        use_bias: bool = False,
        padding: str = "same",
        output_activation: StringOrCallable = "sigmoid",
        dropout_rate: float = 0.0,
        name: str = "convolutional-decoder",
    ):
        self.reshape_into = reshape_into
        self.name = name
        self.filters = make_list_if_not(filters)
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation

        a, b, c = reshape_into
        self.layers = [Dense(a * b * c), Reshape(reshape_into)]

        self.layers.extend(
            _get_conv_transpose_blocks(
                filters,
                kernel_size,
                strides,
                use_bias,
                padding,
                dropout_rate,
                output_activation,
            )
        )

    def __call__(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        return x


class CVAE(Model):
    name = "CVAE"

    def __init__(
        self,
        latent_features: int,
        encoder_filters: ListOrInt,
        decoder_filters: ListOrInt,
        reshape_into: Tuple[int, int, int],
        hidden_activation: StringOrCallable = "relu",
        output_activation: StringOrCallable = "sigmoid",
        kernel_size: TupleOrInt = (5, 5),
        strides: TupleOrInt = (2, 2),
        use_bias: bool = False,
        padding: str = "same",
        dropout_rate: float = 0.0,
    ):
        super(CVAE, self).__init__()
        self.image_shape = latent_features
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.reshape_into = reshape_into
        self.output_activation = output_activation
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.padding = padding
        self.dropout_rate = dropout_rate

        self.encoder = CEncoder(
            latent_features,
            encoder_filters,
            kernel_size,
            strides,
            use_bias,
            padding,
            dropout_rate,
        )

        self.decoder = CDecoder(
            reshape_into,
            decoder_filters,
            kernel_size,
            strides,
            use_bias,
            padding,
            output_activation,
            dropout_rate,
        )

    # a hack for plotting a model
    def _model(self, input_signature: Tuple):
        i = Input(input_signature)
        z, means, log_vars = self.encoder(i)
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
        
        z, mean, logvars = self.encode(inputs, training=False)
        return self.decode(z, training=False)
    
    def encode(self, inputs: TensorOrNDArray) -> ThreeTensors:
        return self.encoder(inputs, training=False)

    def decode(self, inputs: TensorOrNDArray) -> Tensor:
        return self.decoder(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z, means, log_vars = self.encoder(data)
            reconstruction = self.decoder(z)
            total_loss, reconstruction_loss, kl_loss = self.loss(data, [reconstruction, means, log_vars])

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
