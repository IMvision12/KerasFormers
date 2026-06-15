import keras

from kerasformers.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="kerasformers")
class GroundingDinoTokenizer(BaseTokenizer):
    """BERT WordPiece tokenizer for Grounding DINO text prompts.

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given). ``call`` returns ``input_ids`` plus the
    ``attention_mask`` and ``token_type_ids`` the detector consumes.

    Args:
        hf_id: Hub repo to pull ``tokenizer.json`` from.
        tokenizer_file: Explicit path to a ``tokenizer.json``.
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
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
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
