from typing import List, Union, Callable

from tensorflow.keras import layers

from ..base import _ModelExtension


class Autoencoder:
    def __init__(
            self,
            layer_dimensions: List[int],
            output_layer_activation: Union[str, Callable] = None,
            hidden_layer_activation: Union[str, Callable] = "relu",
            dropout_rate: float = 0.0
    ):
        self.dropout_rate = dropout_rate
        self._layer_dimensions = layer_dimensions
        self.output_layer_activation = output_layer_activation
        self.hidden_layer_activation = hidden_layer_activation

        self._hidden_layers = [
            layers.Dense(dim, activation=hidden_layer_activation) for dim in layer_dimensions[:-1]
        ]
        self.output_layer = layers.Dense(layer_dimensions[-1], activation=output_layer_activation)

    def __call__(self, inputs, *args, **kwargs) -> _ModelExtension:
        outputs = inputs
        for layer in self._hidden_layers:
            outputs = layer(outputs)
            if self.dropout_rate > 0:
                outputs = layers.Dropout(self.dropout_rate)(outputs)

        outputs = self.output_layer(outputs)
        return _ModelExtension(inputs, outputs, *args, **kwargs)



