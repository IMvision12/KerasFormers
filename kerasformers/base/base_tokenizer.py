import keras


class BaseTokenizer(keras.layers.Layer):
    """Abstract base for kerasformers tokenizers.

    Subclasses must implement ``call`` (text -> ids) and ``decode``
    (ids -> text). ``batch_decode`` is provided as a pure-Python loop
    over ``decode``.

    Concrete tokenizers add their own state (vocab path, merges,
    special-token ids, BPE / SentencePiece backend) and their own
    ``get_config`` payload — the base intentionally bakes in no
    defaults.
    """

    @classmethod
    def from_weights(cls, identifier, **kwargs):
        """Load the tokenizer to match a model load.

        Mirrors :meth:`BaseModel.from_weights` and takes the *same* ``identifier``,
        so the two stay in lockstep::

            gen = Qwen2Generate.from_weights("qwen2-7b-instruct")
            tok = Qwen2Tokenizer.from_weights("qwen2-7b-instruct")

        - a kerasformers release variant (e.g. ``"qwen2-7b-instruct"``) — the
          official model; dispatches to :meth:`from_release`.
        - ``"hf:org/repo"`` — a community finetune; dispatches to :meth:`from_hf`,
          pulling the tokenizer files from that repo.
        """
        if identifier.startswith("hf:"):
            return cls.from_hf(identifier[len("hf:") :], **kwargs)
        return cls.from_release(identifier, **kwargs)

    @classmethod
    def from_release(cls, variant, /, **kwargs):
        """Build the family-standard tokenizer for an official release ``variant``.

        The official tokenizer is shared across a family's sizes, so the default
        just builds the class default (its constructor pulls the bundled / hub
        assets). Override only if a family needs per-variant resolution.
        """
        return cls(**kwargs)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Build the tokenizer from a Hugging Face repo (a community finetune).

        The default downloads the tokenizer files via the ``hf_id`` constructor
        argument (e.g. ``tokenizer.json``). Families whose tokenizer is assembled
        from other files (e.g. CLIP's ``vocab.json`` + ``merges.txt``) override
        this to fetch them from ``repo``.
        """
        import inspect

        if "hf_id" not in inspect.signature(cls).parameters:
            raise NotImplementedError(
                f"{cls.__name__} cannot load from an 'hf:' repo — its constructor "
                f"takes no `hf_id`. Use a release variant, or override `from_hf` "
                f"to fetch the tokenizer files from {repo!r}."
            )
        return cls(hf_id=repo, **kwargs)

    def __call__(self, *args, **kwargs):
        # Tokenizers are stateless utility layers (no weights to build) and take
        # Python inputs — strings or chat-message lists, not tensors. Forward
        # straight to `call` so they can be passed positionally (Keras's
        # Layer.__call__ rejects non-tensor positional args).
        return self.call(*args, **kwargs)

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
