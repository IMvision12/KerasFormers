import keras
from keras import ops

USE_FUSED_ATTENTION = True


def _softcap_fuses():
    if keras.config.backend() != "jax":
        return False
    try:
        import jax

        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


def fused_attention(query, key, value, scale, attention_mask=None, soft_cap=None):
    """Scaled dot-product attention with an optional fused (flash) backend.

    Computes ``softmax(soft_cap(QKᵀ · scale) + mask) V`` — identical math to the
    manual path the models used before, but routed through
    :func:`keras.ops.dot_product_attention` (which can dispatch to flash /
    cuDNN / XLA-fused kernels) when possible. Falls back to the manual
    implementation when fusion is disabled or unsupported (e.g. logit
    soft-capping off JAX/TPU).

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

    Returns:
        ``(batch, num_heads, q_len, head_dim)``.
    """
    fuse = USE_FUSED_ATTENTION and (soft_cap is None or _softcap_fuses())
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
    return ops.matmul(probs, value)
