import keras

from kerasformers.base import BaseTokenizer

DEFAULT_TOKENIZER_REPO = "openai/gpt-oss-20b"


@keras.saving.register_keras_serializable(package="kerasformers")
class GptOssTokenizer(BaseTokenizer):
    """GPT-OSS tokenizer (``o200k_harmony``, ``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` and exposes ``encode`` / ``decode`` plus
    a ``call`` that tokenizes text(s) or a chat ``messages`` list (rendered with a
    minimal Harmony template) into padded ``{"input_ids", "attention_mask"}``.
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
        # Harmony assistant turns end with <|return|>; fall back to <|endoftext|>.
        self.eos_token = "<|return|>"
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        if self.eos_token_id is None:
            self.eos_token = "<|endoftext|>"
            self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(self, messages, add_generation_prompt=True):
        text = ""
        for m in messages:
            text += f"<|start|>{m['role']}<|message|>{m['content']}<|end|>"
        if add_generation_prompt:
            text += "<|start|>assistant"
        return text

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        input_ids, attention_mask = self.pad_batch([self.encode(t) for t in texts])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
