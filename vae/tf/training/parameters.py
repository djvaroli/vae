from dataclasses import dataclass
from typing import Tuple

from tensorflow.keras.optimizers import Optimizer

from .losses import SigmaVAELoss, VAELoss, _VAELossBase

_vae_type_to_function = {
    "vae": VAELoss,
    "beta-vae": VAELoss,
    "sigma-vae": SigmaVAELoss,
}


@dataclass
class RunParameters:
    latent_dimension: int
    image_shape: Tuple[int, int]
    batch_size: int
    dataset: str
    vae_type: str
    epochs: int
    seed: int
    beta: float = 1.0
    loss_scaling: float = 1e-4
    learning_rate: float = 5e-4
    gen_reference_mean: float = 0.0
    gen_reference_std: float = 1.0
    optimizer_config: dict = None

    def as_dict(self):
        return self.__dict__

    def set_optimizer_config(self, optimizer: Optimizer):
        self.optimizer_config = optimizer.get_config()

    @property
    def loss_fn(self) -> _VAELossBase:
        return _vae_type_to_function[self.vae_type](self.beta, self.loss_scaling)
