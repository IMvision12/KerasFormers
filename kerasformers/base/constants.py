# Additive mask value used to zero out attention logits before softmax: exp(-1e9)
# underflows to 0 in float32, so adding it to disallowed positions removes them.
# Import this wherever a large-negative additive attention / padding mask is built
# (previously the literal -1e9 / a private _MASK_NEG was duplicated per model).
MASK_NEG = -1e9
