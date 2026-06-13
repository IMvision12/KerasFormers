import keras

from kerasformers.base import BaseTokenizer

DEFAULT_TOKENIZER_REPO = "mistralai/Mistral-Large-3-675B-Instruct-2512"


@keras.saving.register_keras_serializable(package="kerasformers")
class Mistral4Tokenizer(BaseTokenizer):
    """Mistral Large 3 Tekken tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given) and exposes ``encode`` / ``decode`` plus a
    ``call`` that tokenizes text(s) or a chat ``messages`` list (rendered with
    the ``[INST]`` template) into padded ``{"input_ids", "attention_mask"}``
    with the ``<s>`` bos prepended.
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
        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(self, messages, add_generation_prompt=True, system=None):
        if messages and messages[0].get("role") == "system":
            system = messages[0]["content"]
            messages = messages[1:]
        text = ""
        for i, m in enumerate(messages):
            if m["role"] == "user":
                content = m["content"]
                if i == 0 and system is not None:
                    content = f"{system}\n\n{content}"
                text += f"[INST] {content} [/INST]"
            else:
                text += f" {m['content']}{self.eos_token}"
        return text

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        input_ids, attention_mask = self.pad_batch(
            [[self.bos_token_id] + self.encode(t) for t in texts]
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
