import numpy as np
from keras import ops

from kerasformers.base.base_mixin import PreprocessorMixin


class BaseTokenizer(PreprocessorMixin):
    """Abstract base for kerasformers tokenizers.

    Subclasses implement ``call`` (text -> ids) and ``decode`` (ids -> text);
    ``batch_decode`` is a pure-Python loop over ``decode``. The loading API
    (``from_weights`` / ``from_release`` / ``from_hf``) and the ``__call__`` ->
    ``call`` forwarder are inherited from :class:`PreprocessorMixin`.

    Shared ``call`` / ``decode`` plumbing is provided as helpers so concrete
    tokenizers keep only their backend-specific encode/decode:

    * :meth:`normalize_texts` — coerce ``call`` input to a list of strings (with
      ChatML-messages dispatch when the subclass defines ``apply_chat_template``);
    * :meth:`pad_batch` — right-pad ragged id sequences to a rectangular
      ``input_ids`` + ``attention_mask``;
    * :meth:`to_id_list` — normalize a ``decode`` argument (tensor / numpy / int /
      list) to a flat list of ints;
    * :meth:`encode_batch_to_inputs` — for ``tokenizers``-padded backends, encode
      a batch (optionally as text pairs) straight to an ``input_ids`` (+ optional
      ``attention_mask`` / ``token_type_ids``) tensor dict.

    Concrete tokenizers add their own state (vocab path, merges, special-token
    ids, BPE / SentencePiece backend) and ``get_config`` payload — the base
    intentionally bakes in no defaults.
    """

    def normalize_texts(self, inputs):
        if inputs is None:
            raise ValueError(f"No text inputs provided to {type(self).__name__}.")
        if (
            isinstance(inputs, (list, tuple))
            and inputs
            and isinstance(inputs[0], dict)
            and "role" in inputs[0]
            and hasattr(self, "apply_chat_template")
        ):
            return [self.apply_chat_template(inputs)]
        return [inputs] if isinstance(inputs, str) else list(inputs)

    def pad_batch(self, sequences, pad_value=0):
        max_len = max((len(s) for s in sequences), default=0)
        input_ids = np.full((len(sequences), max_len), pad_value, dtype=np.int32)
        attention_mask = np.zeros((len(sequences), max_len), dtype=np.int32)
        for i, s in enumerate(sequences):
            input_ids[i, : len(s)] = s
            attention_mask[i, : len(s)] = 1
        return input_ids, attention_mask

    def to_id_list(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return [int(i) for i in ids]

    def encode_batch_to_inputs(
        self, inputs, text_pair=None, token_type_ids=True, mask_dtype="int32"
    ):
        texts = self.normalize_texts(inputs)
        if text_pair is None:
            encs = self._tok.encode_batch(texts)
        else:
            pairs = [text_pair] if isinstance(text_pair, str) else list(text_pair)
            encs = self._tok.encode_batch(list(zip(texts, pairs)))
        out = {"input_ids": ops.convert_to_tensor([e.ids for e in encs], dtype="int32")}
        if mask_dtype is not None:
            out["attention_mask"] = ops.convert_to_tensor(
                [e.attention_mask for e in encs], dtype=mask_dtype
            )
        if token_type_ids:
            out["token_type_ids"] = ops.convert_to_tensor(
                [e.type_ids for e in encs], dtype="int32"
            )
        return out

    def call(self, inputs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(inputs)`."
        )

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement `decode(ids, skip_special_tokens)`."
        )

    def batch_decode(self, ids_batch, skip_special_tokens: bool = True):
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]
