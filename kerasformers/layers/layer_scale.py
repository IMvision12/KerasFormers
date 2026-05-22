import keras
from keras import initializers, layers


@keras.saving.register_keras_serializable(package="kerasformers")
class LayerScale(layers.Layer):
    """
    Implements LayerScale, a learnable scaling layer that multiplies the input by a
    trainable scale factor. It is often used in modern architectures to add stability
    to the training process by scaling the output of certain layers.

    Args:
        layer_scale_init (float): Initial value for the scaling factor `gamma`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Methods:
        build(input_shape):
            Creates the trainable scaling factor `gamma`, initialized to the `layer_scale_init`
            and with the shape automatically determined from the input shape.
        call(x):
            Multiplies the input `x` by the scaling factor `gamma`.
        get_config():
            Returns a dictionary containing the configuration of the layer.

    Example:
        >>> layer = LayerScale(layer_scale_init=0.1)
        >>> output = layer(input_tensor)
    """

    def __init__(self, layer_scale_init, **kwargs):
        super().__init__(**kwargs)
        self.layer_scale_init = layer_scale_init

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer=initializers.Constant(self.layer_scale_init),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_scale_init": self.layer_scale_init,
            }
        )
        return config
