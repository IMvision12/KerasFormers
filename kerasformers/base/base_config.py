"""Typed model configuration base, in the spirit of a transformers config.

A ``BaseConfig`` subclass declares each hyperparameter as an annotated class
attribute with its default, so the config is the single, documented source of a
model's shape. It serializes flat (every field at the top level, plus
``model_type``), which is what a repo's ``kf_config.json`` carries, and rebuilds
from that flat dict, restoring tuples for tuple-typed fields (JSON has no tuple).

Example::

    class DetrConfig(BaseConfig):
        model_type = "detr"
        backbone_variant: str = "ResNet50"
        hidden_dim: int = 256
        num_queries: int = 100

    cfg = DetrConfig(backbone_variant="ResNet101")
    cfg.to_dict()          # {"model_type": "detr", "backbone_variant": "ResNet101", ...}
    DetrConfig.from_dict(cfg.to_dict())   # round-trips, tuples restored
"""

import typing


def _is_tuple_type(annotation):
    """True when ``annotation`` is ``tuple`` / ``tuple[...]`` / ``tuple | None``."""
    if annotation is tuple or typing.get_origin(annotation) is tuple:
        return True
    return any(
        arg is tuple or typing.get_origin(arg) is tuple
        for arg in typing.get_args(annotation)
    )


class BaseConfig:
    """Base for typed, flat-serializing kerasformers model configs."""

    model_type = None

    def __init__(self, **kwargs):
        annotations = self._annotations()
        for name in annotations:
            value = kwargs.pop(name, getattr(type(self), name, None))
            setattr(self, name, self._coerce(name, value, annotations[name]))
        # Keep any unknown kwargs (e.g. fields added by a newer version) so the
        # config still round-trips a repo written by a later release.
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _annotations(cls):
        merged = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "__annotations__", {}) or {})
        return merged

    @classmethod
    def field_names(cls):
        return list(cls._annotations().keys())

    @staticmethod
    def _coerce(name, value, annotation):
        # JSON lists come back as lists; restore tuples for tuple-typed fields.
        if isinstance(value, list) and _is_tuple_type(annotation):
            return tuple(value)
        return value

    def to_dict(self):
        """Flat dict of all fields, ``model_type`` first."""
        data = {}
        if self.model_type is not None:
            data["model_type"] = self.model_type
        for name in self.field_names():
            data[name] = getattr(self, name)
        return data

    def constructor_kwargs(self):
        """Fields to pass to the model constructor (``to_dict`` minus meta)."""
        return {k: v for k, v in self.to_dict().items() if k != "model_type"}

    @classmethod
    def from_dict(cls, data):
        """Build from a flat dict, ignoring any non-field (metadata) keys."""
        fields = set(cls.field_names())
        return cls(**{k: v for k, v in data.items() if k in fields})

    def __repr__(self):
        inner = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"{type(self).__name__}({inner})"
