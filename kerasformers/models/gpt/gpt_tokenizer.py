import keras
import numpy as np

from kerasformers.base import BaseTokenizer

DEFAULT_TOKENIZER_REPO = "openai-community/openai-gpt"


@keras.saving.register_keras_serializable(package="kerasformers")
class GptTokenizer(BaseTokenizer):
    """Original GPT BPE tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` (lowercasing byte-pair encoder with
    ``</w>`` word boundaries) and exposes ``encode`` / ``decode`` and a ``call``
    that tokenizes text(s) into padded ``{"input_ids", "attention_mask"}``. GPT is
    a base LM with no special end-of-text token.
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
        self.eos_token_id = None

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def call(self, inputs):
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        ids = [self.encode(t) for t in texts]
        max_len = max(len(s) for s in ids)
        input_ids = np.zeros((len(ids), max_len), dtype="int32")
        attention_mask = np.zeros((len(ids), max_len), dtype="int32")
        for i, s in enumerate(ids):
            input_ids[i, : len(s)] = s
            attention_mask[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(
            [int(i) for i in ids], skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
