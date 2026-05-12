def copy_weights_by_path_suffix(src, dst):
    """Copy matching weights from ``src`` into ``dst`` by stable path suffix.

    Drops auto-counter wrappers like ``clip_attention_X/`` so weights can be
    shared across classes built with different layer-instantiation orders
    (where Keras' auto-counter would otherwise misalign). Only weights with
    matching last-two path segments *and* identical shape are assigned;
    everything else is left untouched.

    Used to warm-start a task-specific subclass (e.g.
    ``CLIPImageClassify``) from a base model checkpoint
    (``CLIPModel.from_weights(variant)``).
    """

    def _key(w):
        return "/".join(w.path.split("/")[-2:])

    src_map = {_key(w): w for w in src.weights}
    for dst_w in dst.weights:
        src_w = src_map.get(_key(dst_w))
        if src_w is not None and tuple(src_w.shape) == tuple(dst_w.shape):
            dst_w.assign(src_w)
