import keras

from kerasformers.base.base_mixin import WeightLoadingMixin


def hf_num_classes(hf_config):
    """Derive the class count from a ``config.json`` dict.

    A serialized ``config.json`` typically stores ``id2label`` rather than a
    direct count, so this helper derives it from whichever of ``num_labels`` /
    ``id2label`` / ``label2id`` is present.
    """
    if "num_labels" in hf_config:
        return hf_config["num_labels"]
    id2label = hf_config.get("id2label")
    if id2label:
        return len(id2label)
    label2id = hf_config.get("label2id")
    if label2id:
        return len(label2id)
    raise KeyError(
        "Could not determine num_labels from HF config.json — "
        "neither 'num_labels' nor 'id2label' / 'label2id' is present."
    )


class FunctionalBaseModel(WeightLoadingMixin, keras.Model):
    """Base for *functional* kerasformers models (CLIP, ViT, detectors, …) that
    build themselves with ``super().__init__(inputs=..., outputs=...)``."""


class SubclassedBaseModel(WeightLoadingMixin, keras.Model):
    """Base for *imperative / subclassed* kerasformers models (Qwen LLMs & VLMs).

    Deliberately a **separate** ``keras.Model`` subclass from :class:`FunctionalBaseModel`,
    not a subclass of it. When a functional model is built, Keras runs
    ``inject_functional_model_class`` and rewrites the functional base's
    ``__bases__`` from ``keras.Model`` to ``Functional`` (functional models rely
    on this for their subsequent builds). If subclassed models shared that base,
    ``Functional`` would leak into their MRO too — making Keras treat them as
    functional and fail with ``Functional.__init__() missing ... 'inputs' and
    'outputs'`` on construction, or ``'<Model>' object has no attribute
    '_inputs'`` on call. A separate base keeps subclassed models unaffected.
    """

    def build_for_transfer(self):
        """Build every sublayer with a dummy forward, ready for a weight stream.

        Subclassed models build lazily on first call, so nothing has weights
        until they run once. The converted-weight cache reloads a model from a
        serialized *config* (an unbuilt skeleton) and then streams cached tensors
        onto ``self.weights`` — which requires the weights to exist first. This
        runs the minimal forward that materializes them: a length-4 ``input_ids``
        batch, the text-LLM signature. Non-text subclassed models (VLMs, ASR)
        override with a signature-matching dummy input.
        """
        self({"input_ids": keras.ops.zeros((1, 4), dtype="int32")})
