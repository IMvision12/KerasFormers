import keras
from keras import ops

USE_FUSED_ATTENTION = True


def _fused_op_available():
    # keras.ops.dot_product_attention only dispatches to a real flash / efficient
    # kernel on the JAX and PyTorch backends (matching KerasHub). On TensorFlow it
    # lowers to a manual einsum/softmax, so fusing there only adds transpose
    # overhead around the same math -> use the manual path instead.
    backend = keras.config.backend()
    if backend == "jax":
        try:
            from jax.nn import dot_product_attention  # noqa: F401

            return True
        except ImportError:
            return False
    if backend == "torch":
        try:
            from torch.backends.cuda import can_use_flash_attention  # noqa: F401

            return True
        except ImportError:
            return False
    return False


def _softcap_fuses():
    if keras.config.backend() != "jax":
        return False
    try:
        import jax

        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


def fused_attention(
    query,
    key,
    value,
    scale,
    attention_mask=None,
    soft_cap=None,
    dropout=None,
    training=None,
):
    """Scaled dot-product attention with an optional fused (flash) backend.

    Computes ``softmax(soft_cap(QKᵀ · scale) + mask) V`` — identical math to the
    manual path the models used before, but routed through
    :func:`keras.ops.dot_product_attention` (which can dispatch to flash /
    cuDNN / XLA-fused kernels) when possible. Falls back to the manual
    implementation when fusion is disabled or unsupported: the TensorFlow
    backend (no fused kernel there), logit soft-capping off JAX/TPU, or
    training-time attention dropout (the fused op has no dropout argument).

    All tensors are in ``(batch, num_heads, seq, head_dim)`` layout, with the
    key/value heads already repeated to ``num_heads`` (GQA expansion is the
    caller's responsibility, matching the existing code). The result is returned
    in the same layout.

    Args:
        query: ``(batch, num_heads, q_len, head_dim)``.
        key: ``(batch, num_heads, kv_len, head_dim)``.
        value: ``(batch, num_heads, kv_len, head_dim)``.
        scale: Query/key scaling factor (e.g. ``head_dim**-0.5``).
        attention_mask: Additive mask broadcastable to
            ``(batch, num_heads, q_len, kv_len)``, or ``None``.
        soft_cap: Optional tanh logit soft-cap value (e.g. Gemma's ``50.0``);
            ``None`` disables it.
        dropout: Optional ``keras.layers.Dropout`` applied to the attention
            probabilities. Only used during training (and only when its rate is
            positive); at inference it is a no-op so the fused path is taken.
        training: Whether the call is in training mode. When ``True`` and
            ``dropout`` has a positive rate, the manual path is used so the
            dropout can be applied to the probabilities.

    Returns:
        ``(batch, num_heads, q_len, head_dim)``.
    """
    use_dropout = (
        bool(training) and dropout is not None and getattr(dropout, "rate", 0.0) > 0.0
    )
    fuse = (
        USE_FUSED_ATTENTION
        and _fused_op_available()
        and not use_dropout
        and (soft_cap is None or _softcap_fuses())
    )
    if fuse:
        q = ops.transpose(query, (0, 2, 1, 3))
        k = ops.transpose(key, (0, 2, 1, 3))
        v = ops.transpose(value, (0, 2, 1, 3))
        out = ops.dot_product_attention(
            q,
            k,
            v,
            bias=attention_mask,
            scale=scale,
            attn_logits_soft_cap=soft_cap,
        )
        return ops.transpose(out, (0, 2, 1, 3))

    logits = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * scale
    if soft_cap is not None:
        logits = soft_cap * ops.tanh(logits / soft_cap)
    if attention_mask is not None:
        logits = logits + attention_mask
    probs = ops.cast(ops.softmax(ops.cast(logits, "float32"), axis=-1), query.dtype)
    if use_dropout:
        probs = dropout(probs, training=True)
    return ops.matmul(probs, value)
