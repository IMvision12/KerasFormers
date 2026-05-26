import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kerasformers")
class FlattenChoices(layers.Layer):
    """Merge the multiple-choice axis into the batch: ``(B, C, S) -> (B*C, S)``.

    Defining ``compute_output_shape`` keeps the dynamic reshape out of the
    functional-build trace, so it builds on every backend (the JAX backend
    rejects a symbolic ``(-1, None)`` reshape).
    """

    def call(self, inputs):
        return ops.reshape(inputs, (-1, ops.shape(inputs)[-1]))

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


@keras.saving.register_keras_serializable(package="kerasformers")
class UnflattenChoices(layers.Layer):
    """Inverse of :class:`FlattenChoices` for the scores: ``(B*C, 1) -> (B, C)``.

    Args:
        num_choices: Number of choices ``C`` to fold back out of the batch.
    """

    def __init__(self, num_choices, **kwargs):
        super().__init__(**kwargs)
        self.num_choices = num_choices

    def call(self, inputs):
        return ops.reshape(inputs, (-1, self.num_choices))

    def compute_output_shape(self, input_shape):
        return (None, self.num_choices)

    def get_config(self):
        config = super().get_config()
        config.update({"num_choices": self.num_choices})
        return config
