import os

import neptune.new as neptune
from neptune.new.types import File
import numpy as np
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from numpy.random import default_rng
from tensorflow.keras.optimizers import Adam

from ..data import get_directory_iterator
from ..model import CVAE
from ..training import RunParameters, SigmaVAELoss
from ..viz import image_grid


def train(
    latent_dimension: int,
    image_dimension: int,
    batch_size: int,
    beta: float,
    epochs: int,
    seed: int = 25,
    loss_scaling: float = 1e-4,
):
    run = neptune.init(
        project=os.environ["NEPTUNE_PROJECT"],
        api_token=os.environ["NEPTUNE_TOKEN"],
    )

    parameters = RunParameters(
        latent_dimension=latent_dimension,
        image_shape=(image_dimension, image_dimension),
        batch_size=batch_size,
        dataset="simpsonsfaces",
        vae_type="sigma-vae",
        epochs=epochs,
        seed=seed,
        beta=beta,
        loss_scaling=loss_scaling,
    )

    model = CVAE.for_128x128(3, parameters.latent_dimension)
    model.plot(f"{parameters.dataset}_model_config.pdf")
    data_gen = get_directory_iterator(
        f"{parameters.dataset}/",
        target_shape=parameters.image_shape,
        batch_size=parameters.batch_size,
        seed=parameters.seed,
    )

    reconstruction_reference = next(data_gen)[:10]
    img_grid = image_grid(reconstruction_reference, n_rows=2, clip_range=(0., 1.))
    run[f"reconstruction-reference:original:{parameters.dataset}"].upload(File.as_image(img_grid))

    optimizer = Adam(parameters.learning_rate)
    loss = SigmaVAELoss(beta=parameters.beta, scaling=parameters.loss_scaling)
    model.compile(optimizer, loss)
    parameters.set_optimizer_config(optimizer)

    # train the model
    neptune_cbk = NeptuneCallback(run=run, base_namespace="metrics")
    model.fit(data_gen, epochs=parameters.epochs, callbacks=[neptune_cbk])

    # track original / reconstructed images for reference
    encoded, means, logvars = model.encode(reconstruction_reference)
    reconstructions = model.decode(encoded)
    img_grid = image_grid(
        np.concatenate([reconstruction_reference, reconstructions]), n_rows=4, clip_range=(0.0, 1.0),
    )
    run[f"reconstruction-reference:reconstructed:{parameters.dataset}"].upload(File.as_image(img_grid))

    # track the generated images for reference
    rng = default_rng(seed=parameters.seed)
    refernce_inputs = rng.normal(
        parameters.gen_reference_mean, 1.0, size=(20, parameters.latent_dimension)
    )
    generated_images = model.decode(refernce_inputs)
    img_grid = image_grid(
        generated_images, n_rows=4, clip_range=(0.0, 1.0)
    )
    run[f"generated-images:{parameters.dataset}"].upload(File.as_image(img_grid))

    # stop the Neptune run
    run.stop()
