import keras


class BaseProcessor(keras.layers.Layer):
    """Base class for kerasformers multi-modal processors.

    Multi-modal processors compose a :class:`BaseTokenizer` and a
    :class:`BaseImageProcessor` / :class:`BaseAudioFeatureExtractor`
    into one callable. Subclasses set ``self.tokenizer`` /
    ``self.image_processor`` / ``self.feature_extractor`` in
    ``__init__`` and implement ``call`` to dispatch over their
    component(s). ``decode`` / ``batch_decode`` are wired through to
    the tokenizer.
    """

    @classmethod
    def from_weights(cls, identifier, **kwargs):
        """Load the processor to match a model load.

        Mirrors :meth:`BaseModel.from_weights` and takes the *same* ``identifier``,
        so the two stay in lockstep::

            gen  = Qwen2VLGenerate.from_weights("qwen2-vl-7b-instruct")
            proc = Qwen2VLProcessor.from_weights("qwen2-vl-7b-instruct")

        - a kerasformers release variant (e.g. ``"clip_vit_base_16"``) — the
          official model; dispatches to :meth:`from_release`.
        - ``"hf:org/repo"`` — a community finetune; dispatches to :meth:`from_hf`,
          pulling the tokenizer files from that repo.
        """
        if identifier.startswith("hf:"):
            return cls.from_hf(identifier[len("hf:") :], **kwargs)
        return cls.from_release(identifier, **kwargs)

    @classmethod
    def from_release(cls, variant, /, **kwargs):
        """Build the family-standard processor for an official release ``variant``.

        The official tokenizer is shared across a family's sizes, so the default
        just builds the class default (its constructor pulls the bundled / hub
        assets). Override only if a family needs per-variant resolution.
        """
        return cls(**kwargs)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Build the processor from a Hugging Face repo (a community finetune).

        The default loads the tokenizer files via the ``hf_id`` constructor
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
        # Processors are stateless utility layers (no weights to build) and take
        # Python inputs — a conversation / messages / raw images, not tensors.
        # Forward straight to `call` so a `conversation` can be passed
        # positionally (Keras's Layer.__call__ rejects non-tensor positional args).
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement `call`.")

    def decode(self, *args, **kwargs) -> str:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError(
                f"{type(self).__name__}.decode() requires `self.tokenizer` to be set."
            )
        return tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError(
                f"{type(self).__name__}.batch_decode() requires "
                "`self.tokenizer` to be set."
            )
        return tokenizer.batch_decode(*args, **kwargs)
