import os

import neptune.new as neptune
from tensorflow.keras.models import Model
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from numpy.random import default_rng
from tensorflow.keras.optimizers import Adam

from ..data import get_directory_iterator
from ..model import CVAE
from ..training import RunParameters, SigmaVAELoss
from ..callbacks import LogReconstructionCallback, LogGenReferenceCallback


def train(
    latent_dimension: int,
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
        image_shape=(128, 128),
        batch_size=batch_size,
        dataset="simpsonsfaces",
        vae_type="sigma-vae",
        epochs=epochs,
        seed=seed,
        beta=beta,
        loss_scaling=loss_scaling,
    )

    model: Model = CVAE.for_128x128(3, parameters.latent_dimension)
    optimizer = Adam(parameters.learning_rate)
    loss = SigmaVAELoss(beta=parameters.beta, scaling=parameters.loss_scaling)
    model.compile(optimizer, loss)
    parameters.set_optimizer_config(optimizer)
    run["training-parameters"] = parameters.as_dict()

    data_gen = get_directory_iterator(
        f"{parameters.dataset}/",
        target_shape=parameters.image_shape,
        batch_size=parameters.batch_size,
        seed=parameters.seed,
    )

    # init callbacks
    neptune_cbk = NeptuneCallback(run=run, base_namespace="metrics")
    log_reconstruction_cbk = LogReconstructionCallback(
        run, "reconstruction-reference", next(data_gen)[:10]
    )

    rng = default_rng(seed=parameters.seed)
    gen_reference = rng.normal(
        parameters.gen_reference_mean, 1.0, size=(20, parameters.latent_dimension)
    )
    log_reference_cbk = LogGenReferenceCallback(
        run, "generated-reference", gen_reference
    )

    # train model and save weights
    model.fit(
        data_gen,
        epochs=parameters.epochs,
        callbacks=[neptune_cbk, log_reconstruction_cbk, log_reference_cbk],
    )

    model.save_weights(f"weights_{run._short_id}.h5")

    # stop the Neptune run
    run.stop()
