import keras

from kerasformers.base import BaseTokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."


@keras.saving.register_keras_serializable(package="kerasformers")
class LocateAnythingTokenizer(BaseTokenizer):
    """Qwen2.5 BPE tokenizer extended with LocateAnything's grounding tokens.

    Same ``tokenizers`` backend as :class:`Qwen2Tokenizer`, plus the box / ref /
    coordinate special-token ids and a ``parse_boxes`` helper that turns a
    generated id sequence into ``<box>`` quadruples (each coordinate is
    ``coord_start_token_id + v`` for v in [0, 1000]).
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token = "<|im_end|>"
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.image_token = "<IMG_CONTEXT>"
        self.image_token_id = 151665
        self.image_start_token = "<img>"
        self.image_end_token = "</img>"
        self.box_start_token_id = 151668
        self.box_end_token_id = 151669
        self.ref_start_token_id = 151672
        self.ref_end_token_id = 151673
        self.coord_start_token_id = 151677
        self.coord_end_token_id = 152677
        self.none_token_id = 4064
        self.text_mask_token_id = 151676
        self.null_token_id = 152678
        self.switch_token_id = 152679

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(
        self, messages, add_generation_prompt=True, system=DEFAULT_SYSTEM
    ):
        text = ""
        if system is not None and not any(m.get("role") == "system" for m in messages):
            text += f"<|im_start|>system\n{system}<|im_end|>\n"
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        input_ids, attention_mask = self.pad_batch([self.encode(t) for t in texts])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def parse_boxes(self, ids):
        """Extract bounding boxes from a generated id sequence. Returns a list of
        ``[x1, y1, x2, y2]`` in the [0, 1000] grid (divide by 1000 and multiply by
        the image width/height for pixels)."""
        ids = self.to_id_list(ids)
        boxes, coords, inside = [], [], False
        for tid in ids:
            if tid == self.box_start_token_id:
                inside, coords = True, []
            elif tid == self.box_end_token_id:
                if len(coords) == 4:
                    boxes.append(coords)
                inside, coords = False, []
            elif inside and self.coord_start_token_id <= tid <= self.coord_end_token_id:
                coords.append(tid - self.coord_start_token_id)
        return boxes

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
