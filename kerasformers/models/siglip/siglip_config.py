from kerasformers.base import BaseConfig


class SigLIPConfig(BaseConfig):
    r"""Configuration for the SigLIP dual encoder ([`SigLIPModel`] and its heads).

    The defaults describe the SigLIP Base/16 224 model; other variants override the
    vision / text dimensions, patch size, and vocabulary. One `kf_config.json`
    (declaring the canonical [`SigLIPZeroShotClassify`]) sits on each variant's
    repo, and the vision / text / model / zero-shot / classify heads all load from
    it. Fields mirror the model constructor and serialize flat.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the vision tower is built for.
        patch_size (`int`, *optional*, defaults to 16):
            Patch size of the vision encoder.
        vision_hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the vision encoder.
        vision_num_layers (`int`, *optional*, defaults to 12):
            Depth of the vision encoder.
        vision_num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the vision encoder.
        vision_mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the vision encoder.
        vocab_size (`int`, *optional*, defaults to 32000):
            Text tokenizer vocabulary size.
        embed_dim (`int`, *optional*, defaults to 768):
            Shared image-text projection dimension.
        text_hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the text encoder.
        text_num_layers (`int`, *optional*, defaults to 12):
            Depth of the text encoder.
        text_num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the text encoder.
        text_mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the text encoder.
        max_seq_len (`int`, *optional*, defaults to 64):
            Maximum text sequence length.

    Examples:

    ```python
    >>> from kerasformers.models.siglip import SigLIPConfig, SigLIPModel

    >>> configuration = SigLIPConfig()
    >>> model = SigLIPModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "siglip"

    image_size: int = 224
    patch_size: int = 16
    vision_hidden_dim: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vision_mlp_dim: int = 3072
    vocab_size: int = 32000
    embed_dim: int = 768
    text_hidden_dim: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12
    text_mlp_dim: int = 3072
    max_seq_len: int = 64
