import os

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from numpy.random import default_rng
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ..callbacks import LogGenReferenceCallback, LogReconstructionCallback
from ..data import get_directory_iterator
from ..model import ConvAutoEncoder, ConvVAE
from ..training import AutoEncoderLoss, RunParameters, SigmaVAELoss, VAELoss

_autoencoder_map = {"vae": ConvVAE, "sigma-vae": ConvVAE, "ae": ConvAutoEncoder}

_ae_type_loss_map = {
    "vae": VAELoss,
    "sigma-vae": SigmaVAELoss,
    "ae": AutoEncoderLoss
}


def train(
    latent_dimension: int,
    batch_size: int,
    beta: float,
    epochs: int,
    vae_type: str,
    seed: int,
    loss_scaling: float,
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
        vae_type=vae_type,
        epochs=epochs,
        seed=seed,
        beta=beta,
        loss_scaling=loss_scaling,
    )

    architecture_base = _autoencoder_map[vae_type]
    model: Model = architecture_base.for_128x128(3, parameters.latent_dimension)

    optimizer = Adam(parameters.learning_rate)
    loss_base = _ae_type_loss_map[vae_type]
    loss = loss_base(beta=parameters.beta, scaling=parameters.loss_scaling)
    
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
