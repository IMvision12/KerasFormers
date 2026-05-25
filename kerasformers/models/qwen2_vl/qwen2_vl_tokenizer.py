import keras

from kerasformers.base import BaseTokenizer

DEFAULT_TOKENIZER_REPO = "Qwen/Qwen2-VL-2B-Instruct"


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLTokenizer(BaseTokenizer):
    """Qwen2 BPE tokenizer (``tokenizers`` backend).

    Args:
        hf_id: Hub repo to pull ``tokenizer.json`` from.
        tokenizer_file: Explicit path to a ``tokenizer.json`` (overrides the
            download).
    """

    def __init__(self, hf_id=DEFAULT_TOKENIZER_REPO, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import AddedToken, Tokenizer

        if tokenizer_file is None:
            from huggingface_hub import hf_hub_download

            tokenizer_file = hf_hub_download(hf_id, "tokenizer.json")
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)

        for pad_token in ("<|image_pad|>", "<|video_pad|>"):
            if self._tok.token_to_id(pad_token) is None:
                self._tok.add_special_tokens(
                    [AddedToken(pad_token, special=True, normalized=False)]
                )

        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.eos_token = "<|im_end|>"
        self.image_token_id = self._tok.token_to_id(self.image_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        """Text -> list[int] (no auto special tokens; the template carries them)."""
        return self._tok.encode(text, add_special_tokens=False).ids

    def call(self, inputs):
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        ids = [self.encode(t) for t in texts]
        return {"input_ids": ids}

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
