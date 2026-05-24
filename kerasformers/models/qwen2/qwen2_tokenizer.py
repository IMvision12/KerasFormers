import keras
import numpy as np

from kerasformers.base import BaseTokenizer

DEFAULT_TOKENIZER_REPO = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_SYSTEM = "You are a helpful assistant."


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2Tokenizer(BaseTokenizer):
    """Qwen2 BPE tokenizer (``tokenizers`` backend)."""

    def __init__(self, hf_id=DEFAULT_TOKENIZER_REPO, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        if tokenizer_file is None:
            from huggingface_hub import hf_hub_download

            tokenizer_file = hf_hub_download(hf_id, "tokenizer.json")
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token = "<|im_end|>"
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(
        self, messages, add_generation_prompt=True, system=DEFAULT_SYSTEM
    ):
        """Render OpenAI-style ``messages`` to a ChatML prompt string."""
        text = ""
        if system is not None and not any(m.get("role") == "system" for m in messages):
            text += f"<|im_start|>system\n{system}<|im_end|>\n"
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def call(self, inputs):
        """Tokenize text(s) or a chat ``messages`` list into model inputs.

        Accepts a single string, a list of strings, or a ChatML-style
        conversation (``[{"role": ..., "content": ...}, ...]``) — the latter is
        run through :meth:`apply_chat_template` first. Returns padded
        ``{"input_ids", "attention_mask"}`` (numpy) ready for ``model.generate``.
        """
        if (
            isinstance(inputs, (list, tuple))
            and inputs
            and isinstance(inputs[0], dict)
            and "role" in inputs[0]
        ):
            texts = [self.apply_chat_template(inputs)]
        else:
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
