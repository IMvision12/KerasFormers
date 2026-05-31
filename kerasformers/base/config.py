from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    """Base for typed kerasformers model configs.

    A lightweight, optional alternative to passing loose constructor kwargs: a
    per-architecture subclass declares the canonical hyperparameter fields, and
    ``from_hf`` is the single place HF ``config.json`` names are mapped to
    kerasformers names. Models may accept ``config=<ModelConfig>`` *or* the
    equivalent keyword args (a shim builds the config from kwargs), so existing
    call sites keep working. ``from_dict`` ignores unknown keys, which lets a
    model build a config from a loose dict (e.g. a shared release config) without
    the per-class ``kwargs.pop(...)`` of descriptor keys.
    """

    @classmethod
    def from_hf(cls, hf_config):
        raise NotImplementedError(f"{cls.__name__} must implement from_hf(hf_config).")

    @classmethod
    def from_dict(cls, mapping):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in mapping.items() if k in names})

    def to_dict(self):
        return asdict(self)
