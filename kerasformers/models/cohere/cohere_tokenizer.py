import keras

from kerasformers.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="kerasformers")
class CohereTokenizer(BaseTokenizer):
    """Cohere Command-R BPE tokenizer (``tokenizers`` backend, 256k vocab).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given). ``encode`` returns the raw BPE ids; the
    Cohere chat template / specials are applied by the caller.

    Args:
        hf_id: Hub repo to pull ``tokenizer.json`` from, **required** (there is
            no default), mirroring ``AutoTokenizer.from_pretrained`` (e.g.
            ``"CohereLabs/c4ai-command-r-v01"``). Optional only when
            ``tokenizer_file`` is supplied.
        tokenizer_file: Explicit path to a local ``tokenizer.json``; when given,
            nothing is downloaded.
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.bos_token = "<BOS_TOKEN>"
        self.eos_token = "<|END_OF_TURN_TOKEN|>"
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
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
