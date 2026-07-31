from kerasformers.models.siglip.siglip_config import SigLIPConfig


class Siglip2Config(SigLIPConfig):
    r"""Configuration for the SigLIP 2 dual encoder ([`SigLIP2Model`] and heads).

    SigLIP 2 reuses the SigLIP architecture, so this config inherits every field
    from [`SigLIPConfig`]; the defaults differ only in the Gemma-style
    `vocab_size` (256000). One `kf_config.json` (declaring the canonical
    [`SigLIP2ZeroShotClassify`]) sits on each variant's repo, and all SigLIP 2
    heads load from it. Fields serialize flat.

    Examples:

    ```python
    >>> from kerasformers.models.siglip2 import Siglip2Config, SigLIP2Model

    >>> configuration = Siglip2Config()
    >>> model = SigLIP2Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "siglip2"

    vocab_size: int = 256000
