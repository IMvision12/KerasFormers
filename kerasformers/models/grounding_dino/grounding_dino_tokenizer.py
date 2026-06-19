import keras

from kerasformers.base import BaseTokenizer

from .config import GROUNDING_DINO_TOKENIZER_URLS


@keras.saving.register_keras_serializable(package="kerasformers")
class GroundingDinoTokenizer(BaseTokenizer):
    """BERT WordPiece tokenizer for Grounding DINO text prompts.

    Loads ``tokenizer.json`` for ``variant`` from the ``grounding_dino`` release
    tag, or on the fly from ``hf_id``, or from an explicit ``tokenizer_file``.
    ``call`` returns ``input_ids`` plus the ``attention_mask`` and
    ``token_type_ids`` the detector consumes.

    Args:
        variant: Release variant key (default ``"grounding_dino_tiny"``); both
            variants share the same bert-base-uncased tokenizer.
        hf_id: Hub repo to pull ``tokenizer.json`` from (on-the-fly path).
        tokenizer_file: Explicit path to a ``tokenizer.json`` (overrides both).
    """

    TOKENIZER_URLS = GROUNDING_DINO_TOKENIZER_URLS
    DEFAULT_VARIANT = "grounding_dino_tiny"

    def __init__(self, variant=None, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        self.variant = variant or self.DEFAULT_VARIANT
        self.hf_id = hf_id
        if hf_id is not None and tokenizer_file is None:
            tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        else:
            tokenizer_file = self.resolve_tokenizer_json(self.variant, tokenizer_file)
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.cls_token_id = self._tok.token_to_id("[CLS]")
        self.sep_token_id = self._tok.token_to_id("[SEP]")
        self.pad_token_id = self._tok.token_to_id("[PAD]") or 0

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text).ids

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        ids = [self.encode(t) for t in texts]
        max_len = max(len(x) for x in ids)
        input_ids = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids]
        attention_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in ids]
        token_type_ids = [[0] * max_len for _ in ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

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
