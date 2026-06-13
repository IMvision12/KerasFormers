import keras

from kerasformers.base import BaseTokenizer

DEFAULT_TOKENIZER_REPO = "deepseek-ai/DeepSeek-V3"


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekV3Tokenizer(BaseTokenizer):
    """DeepSeek-V3 BPE tokenizer (``tokenizers`` backend, 128k vocab).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given) and exposes ``encode`` / ``decode`` plus
    the DeepSeek special tokens. ``encode`` prepends the BOS id (the
    checkpoints use ``add_bos_token=True``); ``call`` returns unpadded id
    lists.

    Args:
        hf_id: Hub repo to pull ``tokenizer.json`` from.
        tokenizer_file: Explicit path to a ``tokenizer.json`` (overrides the
            download).
    """

    def __init__(self, hf_id=DEFAULT_TOKENIZER_REPO, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        if tokenizer_file is None:
            from huggingface_hub import hf_hub_download

            tokenizer_file = hf_hub_download(hf_id, "tokenizer.json")
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.bos_token = "<｜begin▁of▁sentence｜>"
        self.eos_token = "<｜end▁of▁sentence｜>"
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
