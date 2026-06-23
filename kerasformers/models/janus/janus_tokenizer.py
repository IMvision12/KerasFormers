import keras

from kerasformers.base import BaseTokenizer

from .config import JANUS_TOKENIZER_URLS


@keras.saving.register_keras_serializable(package="kerasformers")
class JanusTokenizer(BaseTokenizer):
    """Janus-Pro BPE tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given) and exposes ``encode`` / ``decode`` plus
    the image special tokens (``<image_placeholder>`` / ``<begin_of_image>`` /
    ``<end_of_image>``). ``encode`` prepends the BOS id (the checkpoints use
    ``add_bos_token=True``); ``call`` returns unpadded id lists — the
    :class:`JanusProcessor` expands image placeholders and pads.

    Args:
        hf_id: Hub repo to pull ``tokenizer.json`` from.
        tokenizer_file: Explicit path to a ``tokenizer.json`` (overrides the
            download).
    """

    TOKENIZER_URLS = JANUS_TOKENIZER_URLS
    DEFAULT_VARIANT = "janus_pro_1b"

    def __init__(self, variant=None, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        if tokenizer_file is None and hf_id is not None:
            tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        else:
            tokenizer_file = self.resolve_tokenizer_json(
                variant or self.DEFAULT_VARIANT, tokenizer_file
            )
        self.variant = variant
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.image_token = "<image_placeholder>"
        self.boi_token = "<begin_of_image>"
        self.eoi_token = "<end_of_image>"
        self.bos_token = "<｜begin▁of▁sentence｜>"
        self.eos_token = "<｜end▁of▁sentence｜>"
        self.image_token_id = self._tok.token_to_id(self.image_token)
        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text, add_bos=True):
        ids = self._tok.encode(text, add_special_tokens=False).ids
        return [self.bos_token_id] + ids if add_bos else ids

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "hf_id": self.hf_id,
                "tokenizer_file": self.tokenizer_file,
            }
        )
        return config
