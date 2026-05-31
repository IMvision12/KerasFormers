import warnings


def warn_random_head(model_cls, head_names):
    """Warn that a task head was left randomly initialized.

    Emitted by task models whose head weights are absent from the loaded
    checkpoint (e.g. a sequence/QnA classifier loaded from a backbone-only
    release). Centralizes the notice that previously lived only in per-class
    docstrings plus one ad-hoc ``warnings.warn``.

    Args:
        model_cls: The task model class (used for the message).
        head_names: A head name or list of head names left at init.
    """
    if isinstance(head_names, str):
        head_names = [head_names]
    names = ", ".join(head_names)
    warnings.warn(
        f"{model_cls.__name__}: task head(s) [{names}] are randomly initialized — "
        f"the loaded checkpoint has no weights for them. Fine-tune before use.",
        stacklevel=2,
    )
