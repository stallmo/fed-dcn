from typing import Protocol, Any

import torch.nn as nn

class AutoencoderFactory(Protocol):
    @staticmethod
    def create_autoencoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> tuple[Any, Any]:
        ...
    @staticmethod
    def create_encoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> nn.Module:
        ...
    @staticmethod
    def create_decoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> nn.Module:
        ...

class StackedAutoencoderFactory:

    @staticmethod
    def create_autoencoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> tuple[Any, Any]:
        encoder = StackedAutoencoderFactory.create_encoder(input_dim, hidden_dims, bottleneck_dim)
        decoder = StackedAutoencoderFactory.create_decoder(input_dim, hidden_dims, bottleneck_dim)
        return encoder, decoder

    @staticmethod
    def create_encoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> nn.Module:
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        encoder = nn.Sequential(*encoder_layers)
        return encoder

    @staticmethod
    def create_decoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> nn.Module:
        decoder_layers = []
        prev_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        decoder = nn.Sequential(*decoder_layers)
        return decoder

class LinearOutputAutoencoderFactory:
    """
    Same stacked fully-connected autoencoder as ``StackedAutoencoderFactory``, but with a linear
    (no ``Sigmoid``) decoder output, for datasets whose reconstruction targets are unbounded
    real-valued vectors (e.g. standardized embeddings) rather than ``[0, 1]``-scaled pixel
    intensities.
    """

    @staticmethod
    def create_autoencoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> tuple[Any, Any]:
        encoder = LinearOutputAutoencoderFactory.create_encoder(input_dim, hidden_dims, bottleneck_dim)
        decoder = LinearOutputAutoencoderFactory.create_decoder(input_dim, hidden_dims, bottleneck_dim)
        return encoder, decoder

    @staticmethod
    def create_encoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> nn.Module:
        return StackedAutoencoderFactory.create_encoder(input_dim, hidden_dims, bottleneck_dim)

    @staticmethod
    def create_decoder(input_dim: int, hidden_dims: list[int], bottleneck_dim: int) -> nn.Module:
        decoder_layers = []
        prev_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder = nn.Sequential(*decoder_layers)
        return decoder
