import keras

from kerasformers.base import BaseTokenizer

DEFAULT_TOKENIZER_REPO = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral3Tokenizer(BaseTokenizer):
    """Mistral 3 Tekken tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given) and exposes ``encode`` / ``decode`` plus
    the Pixtral image special tokens (``[IMG]`` / ``[IMG_BREAK]`` /
    ``[IMG_END]``). ``call`` returns unpadded id lists; the
    :class:`Mistral3Processor` expands image placeholders and pads.

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
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.image_token = "[IMG]"
        self.image_break_token = "[IMG_BREAK]"
        self.image_end_token = "[IMG_END]"
        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.image_token_id = self._tok.token_to_id(self.image_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

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
